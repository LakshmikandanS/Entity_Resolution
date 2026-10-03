"""n03: embed every record of one split with the fine-tuned encoder (GPU, fp16) into on-disk memmaps.

Outputs per source s in {1,2,3} under work/emb/<split>/:
  s<s>_name.npy, s<s>_addr.npy   float16 [n_rows, 128]  (row = data-row index in the source TSV)
  s<s>_flags.npy                 uint8   [n_rows]        bit0 = address empty after light clean, bit1 = Indic name
  s<s>_ids.parquet               entity_id, country per row (written once)
  progress_s<s>.json             rows committed (resume point)

Estimated cost (defaults; train = 12.53M records, test = 11.70M records; two sequences per record):
  VRAM  ~1.2 GB (fp16 weights 0.24 GB + batch 1024 x <=64 tokens activations)
  RAM   <1 GB (one ~160k-row Arrow block, its strings and its embeddings at a time)
  disk  train 12.53M x 2 x 128 x 2 B = 6.4 GB (+ flags 12.5 MB); test 6.0 GB
  time  ~40-70 min per split (estimate: ~400M tokens x 43 MFLOP on GPU + ~25M tokenizer calls on CPU)
Resumable: rows are committed per block; rerun the same command to continue.
Test split: inference only with the frozen train-fitted encoder; no statistic of test is used anywhere.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import (Monitor, add_config_arg, atomic_write_json, clean_address, clean_name, count_rows,
                    has_indic, iter_source, limit_vram, load_config, open_memmap, pick_device, read_json,
                    source_path)
from model import DualFieldEncoder, encode_texts


def embed_unique(model, tok, texts, prefix, max_len, batch, device):
    """Encode each distinct string once (names repeat a lot); returns float16 numpy [len(texts), D]."""
    uniq, inv = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    e = encode_texts(model, tok, [prefix + t for t in uniq.tolist()], max_len, batch, device)
    return e.numpy()[inv], len(uniq)


def ensure_ids(cfg, split, src, path, out: "Path", block):
    if out.exists():
        return
    tmp = out.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, pa.schema([("entity_id", pa.string()), ("country", pa.string())])) as w:
        for _, b in iter_source(path, block):
            w.write_table(pa.table({"entity_id": b.column("entity_id"), "country": b.column("country")}))
    tmp.replace(out)


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--sources", default="1,2,3")
    ap.add_argument("--model-dir", default=None, help="default: work/encoder/final")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec, mc = cfg["embed"], cfg["model"]
    model_dir = cfg.work / "encoder" / "final" if args.model_dir is None else cfg.root / args.model_dir
    if not (model_dir / "meta.json").exists():
        raise SystemExit(f"no trained encoder at {model_dir}; run n02_train_encoder.py first")
    mon = Monitor(cfg, f"n03_embed_{args.split}")
    device = pick_device()
    limit_vram(cfg["train"]["vram_fraction"])
    block = ec["csv_block_bytes"]

    with mon.stage("load_model"):
        model, tok, meta = DualFieldEncoder.load(model_dir)
        model = model.to(device).eval()
        if device.type == "cuda":
            model = model.half()
        dim = meta["proj_dim"]

    out = cfg.wpath("emb", args.split, "x").parent
    for src in [int(s) for s in args.sources.split(",")]:
        path = source_path(cfg, args.split, src)
        n = count_rows(path, cfg.work / "meta")
        with mon.stage(f"ids_s{src}"):
            ensure_ids(cfg, args.split, src, path, out / f"s{src}_ids.parquet", block)
        prog_path = out / f"progress_s{src}.json"
        prog = read_json(prog_path, {"rows_done": 0, "block_bytes": block, "n_rows": n})
        if prog["block_bytes"] != block or prog["n_rows"] != n:
            raise SystemExit(f"{prog_path}: block size or row count changed; delete s{src}_* to restart")
        if prog["rows_done"] >= n:
            print(f"{args.split} s{src}: done ({n:,} rows)")
            continue
        e_name = open_memmap(out / f"s{src}_name.npy", (n, dim), np.float16)
        e_addr = open_memmap(out / f"s{src}_addr.npy", (n, dim), np.float16)
        flags = open_memmap(out / f"s{src}_flags.npy", (n,), np.uint8)
        with mon.stage(f"embed_s{src}"):
            t0, done0, since_flush = time.time(), prog["rows_done"], 0
            for start, b in iter_source(path, block):
                stop = start + b.num_rows
                if stop <= prog["rows_done"]:
                    continue
                names = [clean_name(x) for x in b.column("business_name").to_pylist()]
                addrs = [clean_address(x) for x in b.column("business_address").to_pylist()]
                en, un = embed_unique(model, tok, names, mc["prefix_name"], mc["max_len_name"],
                                      ec["batch_name"], device)
                ea, ua = embed_unique(model, tok, addrs, mc["prefix_addr"], mc["max_len_addr"],
                                      ec["batch_addr"], device)
                e_name[start:stop] = en
                e_addr[start:stop] = ea
                flags[start:stop] = (np.fromiter((a == "" for a in addrs), np.uint8, len(addrs))
                                     | (np.fromiter((has_indic(x) for x in names), np.uint8, len(names)) << 1))
                since_flush += 1
                if since_flush >= ec["flush_every_blocks"] or stop >= n:
                    for mm in (e_name, e_addr, flags):
                        mm.flush()
                    prog["rows_done"] = stop
                    atomic_write_json(prog_path, prog)
                    since_flush = 0
                rate = (stop - done0) / max(time.time() - t0, 1e-6)
                print(f"{args.split} s{src} {stop:,}/{n:,} rows  {rate:,.0f} rows/s  "
                      f"unique {un:,}/{ua:,} of {b.num_rows:,}  eta {(n - stop) / max(rate, 1e-6) / 60:.1f} min  "
                      f"vram_peak {torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0:.2f} GB",
                      flush=True)
            if prog["rows_done"] != n:
                raise RuntimeError(f"{path}: parsed {prog['rows_done']:,} rows but counted {n:,} lines")
        del e_name, e_addr, flags
    mon.close()


if __name__ == "__main__":
    main()
