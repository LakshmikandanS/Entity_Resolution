"""n04: neural pair features for every baseline candidate pair of one split (GPU, blocked, memmapped).

Features (float16 memmap [P, 8] aligned with the baseline pair_idx, work/feats/<split>_neural.npy):
  nn_cos_name, nn_cos_addr, nn_cos_comb (= cosine of normalize(name ⊕ addr) = mean of the two),
  nn_cos_name_if_addr_empty (cos_name when either address is empty, else NaN),
  nn_s1_rank / nn_s1_margin  : rank (1 = best) of cos_comb among the S1's candidates, and best - this,
  nn_t_rank  / nn_t_margin   : the same among all S1s that have this target as a candidate.

Also writes pair metadata memmaps reused by n05/n06 (work/pairs/<split>_{s1_row,t_gid,label}.npy).

Plan:
  pass 0  stream baseline pairs once -> metadata memmaps + per-target-block index files (temp disk P x 16 B)
  pass 1  per block of `target_block_rows` targets: S1 embeddings resident on GPU, target block on GPU,
          cosines in chunks of `pair_chunk` pairs, target-side rank/margin inside the block
  pass 2  stream pairs in S1 order (contiguous per S1) -> S1-side rank/margin

Estimated cost on train (baseline K=32 -> P <= 2.21M x 32 = 70.6M pairs; test P <= 55.4M):
  VRAM  S1 embeddings 2.2M x 256 x 2 B = 1.13 GB + target block 1M x 256 x 2 B = 0.51 GB
        + gathered chunk 1M x 2 x 256 x 2 B = 1.0 GB + sort buffers  ->  ~3 GB peak
  RAM   one block's pair index (~7M x 16 B = 0.11 GB) + metadata chunks  ->  < 1.5 GB
  disk  features P x 8 x 2 B = 1.13 GB; metadata P x 9 B = 0.64 GB; temp P x 16 B = 1.13 GB (deleted)
  time  I/O bound, ~5-15 min per split (18 GFLOP of dot products)
Resumable per pass and per target block (work/feats/<split>_progress.json).
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch

import baseline_api
from common import (Monitor, add_config_arg, atomic_write_json, count_rows, limit_vram, load_config,
                    open_memmap, pick_device, read_json, source_path)

FEATURES = ["nn_cos_name", "nn_cos_addr", "nn_cos_comb", "nn_cos_name_if_addr_empty",
            "nn_s1_rank", "nn_s1_margin", "nn_t_rank", "nn_t_margin"]


def group_rank_margin(group: torch.Tensor, score: torch.Tensor):
    """Within each group id: rank of score (1 = highest, ties broken by position) and (group max - score)."""
    o1 = torch.argsort(score, descending=True, stable=True)
    o2 = torch.argsort(group[o1], stable=True)
    order = o1[o2]
    g = group[order]
    n = g.numel()
    pos = torch.arange(n, device=g.device)
    change = torch.ones(n, dtype=torch.bool, device=g.device)
    if n > 1:
        change[1:] = g[1:] != g[:-1]
    start = torch.cummax(torch.where(change, pos, torch.zeros_like(pos)), 0).values
    rank = torch.empty(n, dtype=torch.float32, device=g.device)
    margin = torch.empty(n, dtype=torch.float32, device=g.device)
    s_sorted = score[order]
    rank[order] = (pos - start + 1).float()
    margin[order] = s_sorted[start] - s_sorted
    return rank, margin


def load_rows_to_gpu(mm: np.ndarray, lo: int, hi: int, device, step=200_000):
    out = torch.empty((hi - lo, mm.shape[1]), dtype=torch.float16, device=device)
    for s in range(lo, hi, step):
        e = min(hi, s + step)
        out[s - lo:e - lo] = torch.from_numpy(np.array(mm[s:e])).to(device)
    return out


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["similarity"]
    split = args.split
    api = baseline_api.load(cfg)
    mon = Monitor(cfg, f"n04_pair_similarity_{split}")
    device = pick_device()
    limit_vram(cfg["train"]["vram_fraction"])

    emb = cfg.work / "emb" / split
    n_s1, n_s2, n_s3 = (count_rows(source_path(cfg, split, s), cfg.work / "meta") for s in (1, 2, 3))
    if (n_s1, n_s2, n_s3) != tuple(api.sizes(split)):
        raise SystemExit(f"row counts {(n_s1, n_s2, n_s3)} differ from the baseline's {api.sizes(split)}")
    for s, n in ((1, n_s1), (2, n_s2), (3, n_s3)):
        if read_json(emb / f"progress_s{s}.json", {}).get("rows_done") != n:
            raise SystemExit(f"embeddings for {split} source {s} incomplete; run n03_embed.py --split {split}")
    mm = {(s, f): np.load(emb / f"s{s}_{f}.npy", mmap_mode="r") for s in (1, 2, 3) for f in ("name", "addr")}
    fl = {s: np.load(emb / f"s{s}_flags.npy", mmap_mode="r") for s in (1, 2, 3)}

    P = api.num_pairs(split)
    n_t = n_s2 + n_s3
    blk = sc["target_block_rows"]
    n_blocks = (n_t + blk - 1) // blk
    feats_path = cfg.wpath("feats", f"{split}_neural.npy")
    prog_path = cfg.work / "feats" / f"{split}_progress.json"
    prog = read_json(prog_path, {"P": P, "pass0": False, "blocks_done": [], "pass2_rows": 0})
    if prog["P"] != P:
        raise SystemExit(f"{prog_path}: pair count changed ({prog['P']} -> {P}); delete work/feats/{split}_* ")
    tmp = cfg.wpath("feats", f"{split}_tmp", "x").parent
    mon.note("scale", {"P": P, "n_s1": n_s1, "n_targets": n_t, "n_blocks": n_blocks,
                       "est_feature_disk_gb": round(P * len(FEATURES) * 2 / 1e9, 2)})

    # ---- pass 0: metadata + per-block pair index -----------------------------------------------------
    pdir = cfg.wpath("pairs", "x").parent
    if not prog["pass0"]:
        with mon.stage("pass0_index"):
            m_s1 = open_memmap(pdir / f"{split}_s1_row.npy", (P,), np.int32)
            m_tg = open_memmap(pdir / f"{split}_t_gid.npy", (P,), np.int32)
            m_lb = open_memmap(pdir / f"{split}_label.npy", (P,), np.int8) if split == "train" else None
            files = [open(tmp / f"b{b}.bin", "wb") for b in range(n_blocks)]
            cols = ["s1_row", "t_src", "t_row"] + (["label"] if split == "train" else [])
            off, last_s1 = 0, -1
            for b in api.iter_pairs(split, cols, sc["pass2_chunk"]):
                k = b["s1_row"].size
                s1 = b["s1_row"].astype(np.int32)
                if s1.size and (s1[0] < last_s1 or np.any(np.diff(s1) < 0)):
                    raise RuntimeError("baseline pairs are not sorted by s1_row (contract violation)")
                last_s1 = int(s1[-1]) if s1.size else last_s1
                gid = np.where(b["t_src"] == 3, b["t_row"].astype(np.int64) + n_s2, b["t_row"]).astype(np.int32)
                m_s1[off:off + k] = s1
                m_tg[off:off + k] = gid
                if m_lb is not None:
                    m_lb[off:off + k] = b["label"]
                pidx = np.arange(off, off + k, dtype=np.int64)
                bid = gid // blk
                for bb in np.unique(bid):
                    sel = bid == bb
                    rec = np.empty(int(sel.sum()), dtype=[("p", "<i8"), ("s", "<i4"), ("t", "<i4")])
                    rec["p"], rec["s"], rec["t"] = pidx[sel], s1[sel], gid[sel] - bb * blk
                    rec.tofile(files[bb])
                off += k
            for f in files:
                f.close()
            if off != P:
                raise RuntimeError(f"iter_pairs yielded {off} pairs, num_pairs says {P}")
            for m in (m_s1, m_tg, m_lb):
                if m is not None:
                    m.flush()
            open_memmap(feats_path, (P, len(FEATURES)), np.float16, fill=np.nan).flush()
            prog["pass0"] = True
            atomic_write_json(prog_path, prog)

    feats = open_memmap(feats_path, (P, len(FEATURES)), np.float16)

    # ---- pass 1: cosines + target-side competition, per target block --------------------------------
    with mon.stage("pass1_blocks"):
        s1n = s1a = None
        s1_empty = torch.from_numpy((np.asarray(fl[1]) & 1).astype(bool)).to(device)
        t_empty_all = np.concatenate([np.asarray(fl[2]) & 1, np.asarray(fl[3]) & 1]).astype(bool)
        for bb in range(n_blocks):
            if bb in prog["blocks_done"]:
                continue
            if s1n is None:
                s1n = load_rows_to_gpu(mm[(1, "name")], 0, n_s1, device)
                s1a = load_rows_to_gpu(mm[(1, "addr")], 0, n_s1, device)
            lo, hi = bb * blk, min(n_t, (bb + 1) * blk)
            parts_n, parts_a = [], []
            for s, base, n in ((2, 0, n_s2), (3, n_s2, n_s3)):
                a, z = max(lo, base), min(hi, base + n)
                if a < z:
                    parts_n.append(load_rows_to_gpu(mm[(s, "name")], a - base, z - base, device))
                    parts_a.append(load_rows_to_gpu(mm[(s, "addr")], a - base, z - base, device))
            tn, ta = torch.cat(parts_n), torch.cat(parts_a)
            te = torch.from_numpy(t_empty_all[lo:hi]).to(device)
            rec = np.fromfile(tmp / f"b{bb}.bin", dtype=[("p", "<i8"), ("s", "<i4"), ("t", "<i4")])
            rec = rec[np.argsort(rec["p"], kind="stable")]
            m = rec.size
            out = torch.empty((m, 4), dtype=torch.float32, device=device)
            for c in range(0, m, sc["pair_chunk"]):
                si = torch.from_numpy(rec["s"][c:c + sc["pair_chunk"]].astype(np.int64)).to(device)
                ti = torch.from_numpy(rec["t"][c:c + sc["pair_chunk"]].astype(np.int64)).to(device)
                cn = (s1n[si] * tn[ti]).sum(-1, dtype=torch.float32)
                ca = (s1a[si] * ta[ti]).sum(-1, dtype=torch.float32)
                empty = s1_empty[si] | te[ti]
                out[c:c + si.numel(), 0] = cn
                out[c:c + si.numel(), 1] = ca
                out[c:c + si.numel(), 2] = (cn + ca) / 2
                out[c:c + si.numel(), 3] = torch.where(empty, cn, torch.full_like(cn, float("nan")))
                del si, ti, cn, ca, empty
            rank, margin = group_rank_margin(torch.from_numpy(rec["t"].astype(np.int64)).to(device), out[:, 2])
            p = rec["p"]
            feats[p, 0:4] = out.cpu().numpy().astype(np.float16)
            feats[p, 6] = rank.clamp_max(65504).cpu().numpy().astype(np.float16)
            feats[p, 7] = margin.cpu().numpy().astype(np.float16)
            feats.flush()
            del tn, ta, te, out, rank, margin
            prog["blocks_done"].append(bb)
            atomic_write_json(prog_path, prog)
            print(f"block {bb + 1}/{n_blocks}: {m:,} pairs", flush=True)
        del s1n, s1a
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- pass 2: S1-side competition (pairs are contiguous per S1) -----------------------------------
    with mon.stage("pass2_s1"):
        m_s1 = np.load(pdir / f"{split}_s1_row.npy", mmap_mode="r")
        r0 = prog["pass2_rows"]
        ch = sc["pass2_chunk"]
        while r0 < P:
            r1 = min(P, r0 + ch)
            if r1 < P:  # extend to the end of the last S1 so no group is split
                last = m_s1[r1 - 1]
                r1 += int(np.searchsorted(np.asarray(m_s1[r1:min(P, r1 + 100_000)]), last, side="right"))
            g = torch.from_numpy(np.asarray(m_s1[r0:r1]).astype(np.int64)).to(device)
            sc_ = torch.from_numpy(np.asarray(feats[r0:r1, 2]).astype(np.float32)).to(device)
            rank, margin = group_rank_margin(g, sc_)
            feats[r0:r1, 4] = rank.clamp_max(65504).cpu().numpy().astype(np.float16)
            feats[r0:r1, 5] = margin.cpu().numpy().astype(np.float16)
            feats.flush()
            r0 = r1
            prog["pass2_rows"] = r0
            atomic_write_json(prog_path, prog)

    atomic_write_json(cfg.work / "feats" / f"{split}_neural.columns.json", FEATURES)
    if not sc["keep_tmp"]:
        shutil.rmtree(tmp, ignore_errors=True)
    mon.close()


if __name__ == "__main__":
    main()
