"""n01: build bi-encoder fine-tuning data from the deterministic 10% encoder split of TRAIN S1.

Positives: ground-truth links of encoder-split S1s. Hard negatives: the baseline's candidate pairs of the same
S1 that are not links, best-ranked first (same blocking / name family, different entity). Text is raw with
light cleaning only (no baseline normalisation); baseline-normalised keys are computed only to mask
same-core-name / same-address twins in the loss.

Reads TRAIN only. Never touches test.

Estimated cost on the full train split (see README "Resource budget"):
  input   S1 210 MB, GT 127 MB, S2+S3 993 MB, pair table scan (~P x 15 B)
  records ~221k S1, ~763k positive links, <=2.2M hard-negative slots, <=3M distinct target records
  RAM     peak ~1.5 GB (Arrow tables of selected rows; no full-file materialisation)
  VRAM    0
  disk    ~0.4-0.6 GB written to work/encoder_data
  time    10-20 min (TSV parsing + baseline key normalisation of ~3M strings)
Not resumable mid-run (single pass); reruns are skipped once work/encoder_data/DONE exists (--force to redo).
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import baseline_api
from common import (Monitor, add_config_arg, atomic_write_json, clean_address, clean_name,
                    encoder_split_mask, encoder_val_mask, explode_ground_truth, ground_truth_path,
                    hash_keys, iter_ground_truth, iter_source, load_config, source_path)


def _in_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if sorted_set.size == 0:
        return np.zeros(values.shape, dtype=bool)
    pos = np.searchsorted(sorted_set, values)
    pos[pos == sorted_set.size] = 0
    return sorted_set[pos] == values


def _keys(fn, texts, countries, chunk=100_000) -> np.ndarray:
    out = []
    for s in range(0, len(texts), chunk):
        out.append(hash_keys(fn(texts[s:s + chunk], countries[s:s + chunk])))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg.wpath("encoder_data", "DONE").parent
    if (out_dir / "DONE").exists() and not args.force:
        print(f"{out_dir} already built; use --force to rebuild")
        return
    api = baseline_api.load(cfg)
    api.require("01-train", "02", "03-train", "04-train", "05-train")
    ec = cfg["encoder_data"]
    block = cfg["embed"]["csv_block_bytes"]
    mon = Monitor(cfg, "n01_build_encoder_data")
    rng = np.random.default_rng(cfg["train"]["seed"])

    # ---- 1. encoder-split S1 records ---------------------------------------------------------------
    with mon.stage("s1_scan"):
        rows, tables, n_s1 = [], [], 0
        for start, b in iter_source(source_path(cfg, "train", 1), block):
            m = encoder_split_mask(b.column("entity_id").to_pylist(), cfg)
            idx = np.nonzero(m)[0]
            rows.append(idx + start)
            tables.append(pa.Table.from_batches([b]).take(pa.array(idx)))
            n_s1 = start + b.num_rows
        s1 = pa.concat_tables(tables)
        s1_rows = np.concatenate(rows).astype(np.int32)  # ascending
        s1_ids = s1.column("entity_id").to_pylist()
        raw_names = s1.column("business_name").to_pylist()
        raw_addrs = s1.column("business_address").to_pylist()
        countries = s1.column("country").to_pylist()
        s1_name_key = _keys(api.name_key, raw_names, countries)
        s1_addr_key = _keys(api.address_key, raw_addrs, countries)
        s1_val = encoder_val_mask(s1_ids, cfg)
        local_of_id = {e: i for i, e in enumerate(s1_ids)}
        print(f"encoder split: {len(s1_ids):,} of {n_s1:,} train S1 ({len(s1_ids) / max(n_s1, 1):.2%})")

    # ---- 2. positives from ground truth --------------------------------------------------------------
    with mon.stage("ground_truth"):
        id_set = pa.array(s1_ids, pa.string())
        ps, pt = [], []
        for b in iter_ground_truth(ground_truth_path(cfg), block):
            a, t = explode_ground_truth(b)
            m = pc.is_in(a, value_set=id_set)
            ps.append(pc.filter(a, m))
            pt.append(pc.filter(t, m))
        pos_s1 = np.fromiter((local_of_id[x] for x in pa.concat_arrays(ps).to_pylist()), dtype=np.int32)
        pos_tid = pa.concat_arrays(pt)
        n_linked_s1 = int(np.unique(pos_s1).size)
        print(f"positives: {len(pos_s1):,} links for {n_linked_s1:,} encoder S1 "
              f"({1 - n_linked_s1 / max(len(s1_ids), 1):.2%} singletons)")

    # ---- 3. hard negatives from the baseline candidate set ------------------------------------------
    with mon.stage("candidates"):
        cols = ["s1_row", "t_src", "t_row", "label"] + (["cand_rank"] if ec.get("use_cand_rank", True) else [])
        loc, src, trow, rank = [], [], [], []
        n_pos_in_cands = 0
        for b in api.iter_pairs("train", cols, ec["batch_rows"]):
            m = _in_sorted(b["s1_row"], s1_rows)
            n_pos_in_cands += int((m & (b["label"] == 1)).sum())
            m &= b["label"] == 0
            if "cand_rank" in b:
                m &= b["cand_rank"] <= ec["candidate_scan_rank"]
            loc.append(np.searchsorted(s1_rows, b["s1_row"][m]).astype(np.int32))
            src.append(b["t_src"][m].astype(np.int8))
            trow.append(b["t_row"][m].astype(np.int32))
            rank.append(b["cand_rank"][m].astype(np.float32) if "cand_rank" in b
                        else rng.random(int(m.sum()), dtype=np.float32))
        loc, src, trow, rank = (np.concatenate(x) for x in (loc, src, trow, rank))
        order = np.lexsort((rng.random(loc.size), rank, loc))  # by S1, then best rank, random ties
        loc, src, trow = loc[order], src[order], trow[order]
        first = np.r_[0, np.flatnonzero(np.diff(loc)) + 1]
        within = np.arange(loc.size) - np.repeat(first, np.diff(np.r_[first, loc.size]))
        keep = within < ec["hard_neg_pool"]
        hn_loc, hn_src, hn_row, hn_slot = loc[keep], src[keep], trow[keep], within[keep]
        recall = n_pos_in_cands / max(len(pos_s1), 1)
        print(f"baseline blocking recall on encoder split: {recall:.4%}; hard-negative slots {hn_loc.size:,}")

    # ---- 4. target texts (positives by id, hard negatives by row) ------------------------------------
    with mon.stage("targets"):
        pos_set = pc.unique(pos_tid)
        parts = []
        for s in (2, 3):
            need = np.unique(hn_row[hn_src == s])
            for start, b in iter_source(source_path(cfg, "train", s), block):
                r = np.arange(start, start + b.num_rows, dtype=np.int32)
                m = _in_sorted(r, need) | pc.is_in(b.column("entity_id"), value_set=pos_set).to_numpy(
                    zero_copy_only=False)
                idx = np.nonzero(m)[0]
                if idx.size:
                    t = pa.Table.from_batches([b]).take(pa.array(idx))
                    t = t.append_column("t_src", pa.array(np.full(idx.size, s, np.int8)))
                    t = t.append_column("t_row", pa.array(r[idx]))
                    parts.append(t)
        tg = pa.concat_tables(parts)
        t_gid = tg.column("t_src").to_numpy().astype(np.int64) << 32 | tg.column("t_row").to_numpy()
        t_order = np.argsort(t_gid)
        t_gid_sorted = t_gid[t_order]
        # hard negatives -> target index
        q = hn_src.astype(np.int64) << 32 | hn_row
        hn_tidx = t_order[np.searchsorted(t_gid_sorted, q)].astype(np.int32)
        # positives -> target index (every GT target exists in the source files)
        pos_tidx = pc.index_in(pos_tid, value_set=tg.column("entity_id")).to_numpy(zero_copy_only=False)
        if np.isnan(pos_tidx.astype(float)).any():
            raise RuntimeError("some ground-truth targets were not found in train S2/S3")
        pos_tidx = pos_tidx.astype(np.int32)
        owner = np.full(tg.num_rows, -1, np.int32)
        owner[pos_tidx] = pos_s1
        t_names_raw = tg.column("business_name").to_pylist()
        t_addrs_raw = tg.column("business_address").to_pylist()
        t_ctry = tg.column("country").to_pylist()
        t_name_key = _keys(api.name_key, t_names_raw, t_ctry)
        t_addr_key = _keys(api.address_key, t_addrs_raw, t_ctry)
        t_addr_clean = [clean_address(a) for a in t_addrs_raw]
        print(f"target records: {tg.num_rows:,}")

    # ---- 5. write -----------------------------------------------------------------------------------
    with mon.stage("write"):
        pool = ec["hard_neg_pool"]
        hardneg = np.full((len(s1_ids), pool), -1, np.int32)
        hardneg[hn_loc, hn_slot] = hn_tidx
        pq.write_table(pa.table({
            "s1_row": s1_rows, "entity_id": s1_ids,
            "name": [clean_name(x) for x in raw_names], "addr": [clean_address(x) for x in raw_addrs],
            "name_key": s1_name_key, "addr_key": s1_addr_key, "is_val": s1_val,
        }), out_dir / "s1.parquet")
        pq.write_table(pa.table({
            "t_src": tg.column("t_src"), "t_row": tg.column("t_row"), "entity_id": tg.column("entity_id"),
            "name": [clean_name(x) for x in t_names_raw], "addr": t_addr_clean,
            "name_key": t_name_key, "addr_key": t_addr_key,
            "addr_empty": np.array([a == "" for a in t_addr_clean]), "owner_s1": owner,
        }), out_dir / "targets.parquet")
        pq.write_table(pa.table({"s1": pos_s1, "pos": pos_tidx, "is_val": s1_val[pos_s1]}),
                       out_dir / "groups.parquet")
        np.save(out_dir / "hardneg.npy", hardneg)
        n_hn = (hardneg >= 0).sum(1)
        stats = {
            "n_train_s1": n_s1, "n_encoder_s1": len(s1_ids), "n_encoder_val_s1": int(s1_val.sum()),
            "n_positive_links": int(len(pos_s1)), "n_linked_encoder_s1": n_linked_s1,
            "n_target_records": tg.num_rows, "blocking_recall_encoder_split": recall,
            "hard_negs_per_s1": {"mean": float(n_hn.mean()), "p05": float(np.percentile(n_hn, 5)),
                                 "zero": int((n_hn == 0).sum())},
            "target_addr_empty_rate": float(np.mean([a == "" for a in t_addr_clean])),
        }
        atomic_write_json(out_dir / "stats.json", stats)
        (out_dir / "DONE").write_text("ok")
        print(stats)
    mon.close()


if __name__ == "__main__":
    main()
