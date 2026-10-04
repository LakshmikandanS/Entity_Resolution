"""Stage 02 (train only): ground-truth label arrays + the learned Indic rewrite tables.

Labels (work/train/labels/):
  owner.npy      int32[#targets]  S1 row that owns each target, -1 if unlinked (a target has <= 1 owner)
  n_true.npy     int16[#S1]       number of true links per S1 (0 = singleton)
  s1_bucket.npy  uint16[#S1]      crc32(entity_id) % 1000 -> fold = bucket % N_FOLDS, entity-level
Target index convention: S2 rows first (0 .. n2-1), then S3 rows (n2 .. n2+n3-1).

Rewrite tables (artifacts/rewrite_map.json), learned from TRAINING links only (problem.md 3.2):
  name_tokens : position-aligned (transliterated target token -> S1 token) over links whose target
                name is in Indic script and has the same token count as the S1 name; a rule is kept if
                seen >= --min-count times, the dominant mapping has share >= --min-share, and the token
                does not usually map to itself.
  state_parts : (country -> transliterated Indic address part -> S1 state) with the same thresholds.
"""
import argparse
import collections
import glob
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.gpu import check_ram  # noqa: E402
from utils.io import (add_common_args, begin_stage, fail, file_stamp, iter_raw_tsv, limit_threads, log,  # noqa: E402
                      paths_from_args, read_manifest, upstream_stamp, write_json, write_manifest)
from utils.normalization import F_INDIC_NAME, Canonicalizer, country_key  # noqa: E402


def load_lookup(norm_dir, src):
    return (np.load(os.path.join(norm_dir, f"{src}_ids_sorted.npy")),
            np.load(os.path.join(norm_dir, f"{src}_ids_order.npy")))


def resolve(ids_num, lookup, what):
    ids_sorted, order = lookup
    pos = np.searchsorted(ids_sorted, ids_num)
    pos = np.minimum(pos, len(ids_sorted) - 1)
    bad = ids_sorted[pos] != ids_num
    if bad.any():
        fail(f"{int(bad.sum())} {what} ids in the ground truth are missing from the source files")
    return order[pos]


def parts(norm_dir, src):
    return sorted(glob.glob(os.path.join(norm_dir, f"{src}-part-*.parquet")))


def build_labels(paths, man, chunk_rows):
    n1 = man["s1"]["rows"]
    n2, n3 = man["s2"]["rows"], man["s3"]["rows"]
    lk = {s: load_lookup(paths.normalized, s) for s in ("s1", "s2", "s3")}
    owner = np.full(n2 + n3, -1, np.int32)
    n_true = np.zeros(n1, np.int16)
    gt_rows = links = conflicts = 0
    for batch in iter_raw_tsv(paths.ground_truth, chunk_rows):
        s1_ids = batch.column("source1_entity_id").to_pylist()
        lists = batch.column("matched_entity_ids").to_pylist()
        s1_rows = resolve(np.array([int(s[3:]) for s in s1_ids], np.int64), lk["s1"], "S1")
        rep, t2, t3, r2, r3 = [], [], [], [], []
        for r, lst in zip(s1_rows, lists):
            if not lst:
                continue
            for tid in lst.split(","):
                tid = tid.strip()
                if tid.startswith("S2-"):
                    t2.append(int(tid[3:])); r2.append(r)
                elif tid.startswith("S3-"):
                    t3.append(int(tid[3:])); r3.append(r)
                else:
                    fail(f"unexpected id {tid!r} in ground truth")
        tg = np.concatenate([resolve(np.array(t2, np.int64), lk["s2"], "S2") if t2 else np.zeros(0, np.int32),
                             n2 + resolve(np.array(t3, np.int64), lk["s3"], "S3") if t3 else np.zeros(0, np.int32)])
        sr = np.array(r2 + r3, dtype=np.int32)
        conflicts += int((owner[tg] >= 0).sum()) + int(len(tg) - len(np.unique(tg)))
        owner[tg] = sr
        np.add.at(n_true, sr, 1)
        gt_rows += batch.num_rows
        links += len(tg)
    if gt_rows != n1:
        fail(f"ground truth has {gt_rows} rows but S1 has {n1}")
    if conflicts:
        fail(f"{conflicts} targets claimed by more than one S1; exclusivity assumption violated")
    log(f"labels: {gt_rows:,} S1 rows, {links:,} links, singletons {(n_true == 0).mean():.2%}, "
        f"unlinked targets {(owner < 0).mean():.2%}")

    bucket = np.zeros(n1, np.uint16)
    for p in parts(paths.normalized, "s1"):
        t = pq.read_table(p, columns=["row", "entity_id"])
        rows = t.column("row").to_numpy()
        bucket[rows] = [zlib.crc32(e.encode()) % 1000 for e in t.column("entity_id").to_pylist()]
    np.save(os.path.join(paths.labels, "owner.npy"), owner)
    np.save(os.path.join(paths.labels, "n_true.npy"), n_true)
    np.save(os.path.join(paths.labels, "s1_bucket.npy"), bucket)
    return owner, {"links": links, "s1": n1, "targets": n2 + n3, "n2": n2, "n3": n3,
                   "singleton_rate": float((n_true == 0).mean()),
                   "unlinked_target_rate": float((owner < 0).mean()),
                   "matches_per_s1_hist": np.bincount(n_true).tolist()}


def learn_rewrites(paths, man, owner, min_count, min_share):
    s1_parts = parts(paths.normalized, "s1")
    s1_tab = pa.concat_tables([pq.read_table(p, columns=["name_basic", "addr_basic", "country"]) for p in s1_parts])
    canon = Canonicalizer()
    tok_pairs, tok_total = collections.Counter(), collections.Counter()
    part_pairs, part_total = collections.Counter(), collections.Counter()
    n_links = n_aligned = 0
    offset = {"s2": 0, "s3": man["s2"]["rows"]}
    for src in ("s2", "s3"):
        for p in parts(paths.normalized, src):
            t = pq.read_table(p, columns=["row", "name_basic", "addr_indic_parts", "country", "flags"])
            rows = t.column("row").to_numpy().astype(np.int64) + offset[src]
            own = owner[rows]
            linked = own >= 0
            flags = t.column("flags").to_numpy()
            has_ip = pc.not_equal(t.column("addr_indic_parts"), "").to_numpy(zero_copy_only=False)
            sel = np.flatnonzero(linked & (((flags & F_INDIC_NAME) != 0) | has_ip))
            if len(sel) == 0:
                continue
            sub = t.take(pa.array(sel))
            s1 = s1_tab.take(pa.array(own[sel]))
            for tn, ip, ctry, fl, sn, sa, sc in zip(
                    sub.column("name_basic").to_pylist(), sub.column("addr_indic_parts").to_pylist(),
                    sub.column("country").to_pylist(), flags[sel], s1.column("name_basic").to_pylist(),
                    s1.column("addr_basic").to_pylist(), s1.column("country").to_pylist()):
                if fl & F_INDIC_NAME:
                    n_links += 1
                    tt, st = tn.split(), sn.split()
                    if len(tt) == len(st):
                        n_aligned += 1
                        for a, b in zip(tt, st):
                            tok_pairs[(a, b)] += 1
                            tok_total[a] += 1
                if ip:
                    ck = country_key(sc)
                    s1_parts_set = set(sa.split("|"))
                    state = canon.state_of_parts(sa, sc)
                    for part in ip.split("|"):
                        if part and part not in s1_parts_set and state:
                            part_pairs[(ck, part, state)] += 1
                            part_total[(ck, part)] += 1
        log(f"rewrite: scanned {src}, {n_links:,} Indic-name links so far, {len(tok_pairs):,} token pairs")

    best = {}
    for (a, b), c in tok_pairs.items():
        if c > best.get(a, (None, 0))[1]:
            best[a] = (b, c)
    name_rules = {a: b for a, (b, c) in best.items()
                  if b != a and tok_total[a] >= min_count and c / tok_total[a] >= min_share}
    best_p = {}
    for (ck, part, st), c in part_pairs.items():
        if c > best_p.get((ck, part), (None, 0))[1]:
            best_p[(ck, part)] = (st, c)
    state_rules = collections.defaultdict(dict)
    for (ck, part), (st, c) in best_p.items():
        if part_total[(ck, part)] >= min_count and c / part_total[(ck, part)] >= min_share:
            state_rules[ck][part] = st
    stats = {"indic_name_links": n_links, "aligned_links": n_aligned,
             "aligned_share": n_aligned / max(n_links, 1), "name_rules": len(name_rules),
             "state_part_rules": {k: len(v) for k, v in state_rules.items()},
             "top_name_rules": sorted(name_rules.items(), key=lambda kv: -tok_total[kv[0]])[:40]}
    return {"name_tokens": name_rules, "state_parts": dict(state_rules)}, stats


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--chunk-rows", type=int, default=config.CHUNK_ROWS)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-share", type=float, default=0.5)
    args = ap.parse_args()
    args.split = "train"
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    man = read_manifest(paths.normalized, "stage 01 (train)")
    sig = {"stage": "02", "normalized": upstream_stamp(paths.normalized), "gt": file_stamp(paths.ground_truth),
           "min_count": args.min_count, "min_share": args.min_share, "n_folds": config.N_FOLDS}
    rebuild = args.force or not os.path.exists(paths.artifact("rewrite_map.json"))
    if begin_stage(paths.labels, sig, rebuild):
        return
    check_ram(0.8, "02 labels + rewrite map", args.force)
    owner, lstats = build_labels(paths, man, args.chunk_rows)
    rmap, rstats = learn_rewrites(paths, man, owner, args.min_count, args.min_share)
    rmap["params"] = {"min_count": args.min_count, "min_share": args.min_share, "learned_from": "train"}
    write_json(paths.artifact("rewrite_map.json"), rmap)
    log(f"rewrite map: {rstats['name_rules']} name rules, state-part rules {rstats['state_part_rules']}, "
        f"aligned share {rstats['aligned_share']:.3f}")
    log(f"  top rules: {rstats['top_name_rules'][:15]}", mem=False)
    write_manifest(paths.labels, {"labels": lstats, "rewrite": rstats, "n_folds": config.N_FOLDS})


if __name__ == "__main__":
    main()
