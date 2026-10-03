"""Stage 03: canonicalisation -> fixed-width record arrays -> blocking keys -> CSR inverted index.

Outputs (work/<split>/):
  records/s1.npy, records/tgt.npy      structured REC_DTYPE memmaps (targets: S2 rows then S3 rows)
  index/s1_keys.u64, s1_key_offs.npy   S1 query keys grouped by S1 row
  index/{keys,offs,post,fam,idf}.npy   global CSR over target keys, frequency-capped per family
The rewrite map comes from stage 02 (training data) for BOTH splits; nothing is learned from test.
Key document frequencies of the test index are computed on the test targets themselves: that is
unsupervised candidate generation (no labels, no tuning), the same procedure run on new data.
"""
import argparse
import gc
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.blocking import (FAMILIES, MAX_KEYS_PER_RECORD, PartitionedKeyWriter, build_partition_csr,  # noqa: E402
                            family_caps, record_keys)
from utils.gpu import check_ram  # noqa: E402
from utils.io import (add_common_args, fail, limit_threads, log, paths_from_args, read_json,  # noqa: E402
                      read_manifest, stage_done, write_json, write_manifest)
from utils.normalization import F_INDIC_NAME, F_LANDMARK, F_LEGAL_FRONT, Canonicalizer, country_key  # noqa: E402
from utils.records import REC_DTYPE, build_record_block  # noqa: E402

COLS = ["row", "country", "name_basic", "addr_basic", "flags"]


def canon_chunk(canon, table, stats):
    rows, keys_per_rec = [], []
    rw = canon.name_rewrite
    for country, nb, ab, fl in zip(table.column("country").to_pylist(), table.column("name_basic").to_pylist(),
                                   table.column("addr_basic").to_pylist(), table.column("flags").to_numpy()):
        fl = int(fl)
        tokens, core, legal, legal_front = canon.canon_name(nb, fl)
        a = canon.canon_address(ab, country)
        if legal_front:
            fl |= F_LEGAL_FRONT
        if a["landmark"]:
            fl |= F_LANDMARK
        if fl & F_INDIC_NAME:
            stats["indic_names"] += 1
            n_rw = sum(1 for t in nb.split() if t in rw)
            stats["indic_names_rewritten"] += n_rw > 0
            stats["tokens_rewritten"] += n_rw
        stats["legal_front"] += legal_front
        stats["with_legal"] += bool(legal)
        stats["with_state"] += bool(a["state"])
        stats["with_hnum"] += bool(a["hnums"])
        stats["with_skey"] += bool(a["skeys"])
        stats["landmark"] += a["landmark"]
        stats["empty_core"] += not core
        rows.append({"canon": tokens, "core": core, "legal": legal, "tokens": a["tokens"],
                     "hnums": a["hnums"], "skeys": a["skeys"], "state": a["state"],
                     "sorted_str": a["sorted_str"], "flags": fl})
        keys_per_rec.append(record_keys(country_key(country), core, a["hnums"], a["skeys"], a["idwords"]))
        if len(stats["examples"]) < 8:
            stats["examples"].append({"name_basic": nb, "core": core, "legal": legal, "addr_basic": ab,
                                      "addr_tokens": a["tokens"], "hnums": a["hnums"], "skeys": a["skeys"],
                                      "state": a["state"], "idwords": a["idwords"]})
    return rows, keys_per_rec


def keys_to_arrays(keys_per_rec):
    n = sum(len(k) for k in keys_per_rec)
    keys = np.empty(n, np.uint64)
    fam = np.empty(n, np.uint8)
    counts = np.empty(len(keys_per_rec), np.int64)
    i = 0
    for r, kl in enumerate(keys_per_rec):
        counts[r] = len(kl)
        for k, f in kl:
            keys[i] = k
            fam[i] = f
            i += 1
    return keys, fam, counts


def build_records_and_keys(paths, man, canon, chunk_rows, n_partitions):
    n1 = man["s1"]["rows"]
    n2, n3 = man["s2"]["rows"], man["s3"]["rows"]
    rec_s1 = np.lib.format.open_memmap(os.path.join(paths.records, "s1.npy.tmp"), mode="w+",
                                       dtype=REC_DTYPE, shape=(n1,))
    rec_t = np.lib.format.open_memmap(os.path.join(paths.records, "tgt.npy.tmp"), mode="w+",
                                      dtype=REC_DTYPE, shape=(n2 + n3,))
    idx_dir = paths.index
    for p in glob.glob(os.path.join(idx_dir, "tmpkeys-*")):
        os.remove(p)
    writer = PartitionedKeyWriter(idx_dir, n_partitions)
    s1_counts = np.zeros(n1, np.int64)
    stats = {}
    t0 = time.time()
    with open(os.path.join(idx_dir, "s1_keys.u64.tmp"), "wb") as fq:
        for src, rec, offset in (("s1", rec_s1, 0), ("s2", rec_t, 0), ("s3", rec_t, n2)):
            st = {k: 0 for k in ("indic_names", "indic_names_rewritten", "tokens_rewritten", "legal_front",
                                 "with_legal", "with_state", "with_hnum", "with_skey", "landmark",
                                 "empty_core")}
            st["examples"] = []
            st["keys_per_family"] = [0] * len(FAMILIES)
            done = 0
            for part in sorted(glob.glob(os.path.join(paths.normalized, f"{src}-part-*.parquet"))):
                pf = pq.ParquetFile(part)
                for batch in pf.iter_batches(batch_size=chunk_rows, columns=COLS):
                    rows_idx = batch.column(0).to_numpy().astype(np.int64)
                    rows, kpr = canon_chunk(canon, batch, st)
                    block = build_record_block(rows)
                    lo = int(rows_idx[0]) + offset
                    if not np.array_equal(rows_idx, np.arange(rows_idx[0], rows_idx[0] + len(rows_idx))):
                        fail(f"{part}: rows are not contiguous")
                    rec[lo:lo + len(block)] = block
                    keys, fam, counts = keys_to_arrays(kpr)
                    st["keys_per_family"] = (np.array(st["keys_per_family"]) +
                                             np.bincount(fam, minlength=len(FAMILIES))).tolist()
                    if src == "s1":
                        keys.tofile(fq)
                        s1_counts[rows_idx] = counts
                    else:
                        tidx = np.repeat(np.arange(lo, lo + len(block), dtype=np.int32), counts)
                        writer.write(keys, tidx, fam)
                    done += len(block)
                log(f"{paths.split}/{src}: canonicalised {done:,} records ({time.time() - t0:.0f}s)")
            st["records"] = done
            stats[src] = st
    writer.close()
    rec_s1.flush(); rec_t.flush()
    # Drop every reference to the memmaps (the loop variable `rec` still holds rec_t) so Windows
    # releases the file handles before the rename.
    del rec_s1, rec_t, rec
    gc.collect()
    os.replace(os.path.join(paths.records, "s1.npy.tmp"), os.path.join(paths.records, "s1.npy"))
    os.replace(os.path.join(paths.records, "tgt.npy.tmp"), os.path.join(paths.records, "tgt.npy"))
    os.replace(os.path.join(idx_dir, "s1_keys.u64.tmp"), os.path.join(idx_dir, "s1_keys.u64"))
    np.save(os.path.join(idx_dir, "s1_key_offs.npy"), np.r_[0, np.cumsum(s1_counts)].astype(np.int64))
    stats["target_postings_spilled"] = writer.count
    return stats


def build_index(paths, n_targets, n_partitions, caps):
    idx_dir = paths.index
    writer = PartitionedKeyWriter.__new__(PartitionedKeyWriter)
    writer.dir, writer.n = idx_dir, n_partitions
    fam_stats = {f: {"keys": 0, "postings": 0, "keys_dropped": 0, "postings_dropped": 0,
                     "df_hist": [0] * 21} for f in FAMILIES}
    part_files = []
    for p in range(n_partitions):
        csr, st = build_partition_csr(writer.partition_paths(p), caps)
        for f, s in st.items():
            for k in ("keys", "postings", "keys_dropped", "postings_dropped"):
                fam_stats[f][k] += s[k]
            fam_stats[f]["df_hist"] = (np.array(fam_stats[f]["df_hist"]) + np.array(s["df_hist"])).tolist()
        base = os.path.join(idx_dir, f"tmpcsr-{p:03d}")
        for name in ("keys", "df", "fam", "post"):
            np.save(f"{base}.{name}.npy", csr[name])
        part_files.append(base)
        log(f"index partition {p + 1}/{n_partitions}: {len(csr['keys']):,} keys, {len(csr['post']):,} postings")
        del csr
    n_keys = sum(len(np.load(f"{b}.keys.npy", mmap_mode="r")) for b in part_files)
    n_post = sum(len(np.load(f"{b}.post.npy", mmap_mode="r")) for b in part_files)
    out = {name: np.lib.format.open_memmap(os.path.join(idx_dir, f"{name}.npy"), mode="w+", dtype=dt, shape=(n,))
           for name, dt, n in (("keys", np.uint64, n_keys), ("offs", np.int64, n_keys + 1),
                               ("post", np.int32, n_post), ("fam", np.uint8, n_keys), ("idf", np.float32, n_keys))}
    ki = pi = 0
    out["offs"][0] = 0
    for b in part_files:
        k = np.load(f"{b}.keys.npy")
        df = np.load(f"{b}.df.npy")
        out["keys"][ki:ki + len(k)] = k
        out["fam"][ki:ki + len(k)] = np.load(f"{b}.fam.npy")
        out["idf"][ki:ki + len(k)] = np.log1p(n_targets / np.maximum(df, 1)).astype(np.float32)
        out["offs"][ki + 1:ki + 1 + len(k)] = pi + np.cumsum(df)
        post = np.load(f"{b}.post.npy")
        out["post"][pi:pi + len(post)] = post
        ki += len(k)
        pi += len(post)
        for name in ("keys", "df", "fam", "post"):
            os.remove(f"{b}.{name}.npy")
    for a in out.values():
        a.flush()
    keys = out["keys"]
    if n_keys > 1 and not bool(np.all(keys[1:] > keys[:-1])):
        fail("index keys are not strictly increasing (partition concatenation bug)")
    for p in glob.glob(os.path.join(idx_dir, "tmpkeys-*")):
        os.remove(p)
    return {"keys": n_keys, "postings": n_post, "families": fam_stats}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--chunk-rows", type=int, default=config.CHUNK_ROWS)
    ap.add_argument("--partitions", type=int, default=config.INDEX_PARTITIONS)
    ap.add_argument("--max-key-frequency", type=int, default=config.MAX_KEY_FREQUENCY)
    ap.add_argument("--max-key-frequency-common", type=int, default=config.MAX_KEY_FREQUENCY_COMMON)
    args = ap.parse_args()
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    if stage_done(paths.index) and not args.force:
        log(f"{paths.index} already finished (use --force to rebuild)")
        return
    man = read_manifest(paths.normalized, f"stage 01 ({args.split})")
    rmap_path = paths.artifact("rewrite_map.json")
    if not os.path.exists(rmap_path):
        fail("artifacts/rewrite_map.json missing: run 02_build_rewrite_map.py on the training split first")
    rmap = read_json(rmap_path)
    if rmap.get("params", {}).get("learned_from") != "train":
        fail("rewrite map was not learned from the training split")
    canon = Canonicalizer(rmap)
    n_targets = man["s2"]["rows"] + man["s3"]["rows"]
    est_keys = n_targets * 13 / args.partitions
    check_ram(0.4 + args.chunk_rows * MAX_KEYS_PER_RECORD * 24 / 2**30 + 2 * est_keys * 32 / 2**30,
              "03 records + index", args.force)

    marker = os.path.join(paths.index, "_records_done.json")
    if os.path.exists(marker) and not args.force:
        rstats = read_json(marker)
        log("records and keys already built; rebuilding only the CSR index")
    else:
        rstats = build_records_and_keys(paths, man, canon, args.chunk_rows, args.partitions)
        write_json(marker, rstats)
    for src in ("s1", "s2", "s3"):
        s = rstats[src]
        log(f"{src}: {s['records']:,} records; indic names {s['indic_names']:,} "
            f"(rewritten {s['indic_names_rewritten']:,}, tokens {s['tokens_rewritten']:,}); "
            f"state {s['with_state'] / max(s['records'], 1):.1%}, hnum {s['with_hnum'] / max(s['records'], 1):.1%}; "
            f"keys/family {dict(zip(FAMILIES, s['keys_per_family']))}")
    caps = family_caps(args.max_key_frequency, args.max_key_frequency_common)
    istats = build_index(paths, n_targets, args.partitions, caps)
    for f, s in istats["families"].items():
        log(f"family {f}: {s['keys']:,} keys / {s['postings']:,} postings; dropped by cap "
            f"{s['keys_dropped']:,} keys / {s['postings_dropped']:,} postings", mem=False)
    write_manifest(paths.index, {"split": args.split, "records": rstats, "index": istats,
                                 "caps": dict(zip(FAMILIES, caps.tolist())), "partitions": args.partitions,
                                 "n_s1": man["s1"]["rows"], "n_targets": n_targets,
                                 "n2": man["s2"]["rows"], "n3": man["s3"]["rows"]})


if __name__ == "__main__":
    main()
