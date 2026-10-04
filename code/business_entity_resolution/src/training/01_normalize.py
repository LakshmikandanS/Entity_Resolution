"""Stage 01: stream raw TSVs and apply the basic (non-learned) normalisation.

Output per source: work/<split>/normalized/<src>-part-XXXXX.parquet with
  row (int32, row index within the source), id_num (int64), entity_id, country,
  name_basic, addr_basic ('|'-joined parts), addr_indic_parts, flags (uint16)
plus <src>_ids_sorted.npy / <src>_ids_order.npy for id -> row lookup (no Python dicts).
Resumable per part; one source file open at a time; pandas is not used.
"""
import argparse
import collections
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.gpu import check_ram  # noqa: E402
from utils.io import (AtomicParquetWriter, SOURCES, add_common_args, fail, iter_raw_tsv,  # noqa: E402
                      limit_threads, log, numeric_ids, paths_from_args, write_manifest, begin_stage, file_stamp)
from utils.normalization import (F_ADDR_EMPTY, F_ADDR_PLACEHOLDER, F_ALLCAPS, F_ACCENTED,  # noqa: E402
                                 F_DOMAIN, F_INDIC_ADDR, F_INDIC_NAME, basic_address, basic_name)

SCHEMA = pa.schema([
    ("row", pa.int32()), ("id_num", pa.int64()), ("entity_id", pa.string()), ("country", pa.string()),
    ("name_basic", pa.string()), ("addr_basic", pa.string()), ("addr_indic_parts", pa.string()),
    ("flags", pa.uint16()),
])
FLAG_NAMES = {"indic_name": F_INDIC_NAME, "indic_addr": F_INDIC_ADDR, "domain": F_DOMAIN,
              "allcaps_name": F_ALLCAPS, "accented_name": F_ACCENTED, "addr_empty": F_ADDR_EMPTY,
              "addr_placeholder": F_ADDR_PLACEHOLDER}


def normalize_source(paths, src, chunk_rows, force):
    out_dir = paths.normalized
    prefix = "S" + src[-1] + "-"
    raw = paths.raw(src)
    if not os.path.exists(raw):
        fail(f"missing input {raw}")
    if force:
        for p in glob.glob(os.path.join(out_dir, f"{src}-part-*.parquet")):
            os.remove(p)
    stats = collections.Counter()
    countries = collections.Counter()
    examples, indic_examples = [], []
    row0, part = 0, 0
    t0 = time.time()
    for batch in iter_raw_tsv(raw, chunk_rows):
        n = batch.num_rows
        dest = os.path.join(out_dir, f"{src}-part-{part:05d}.parquet")
        if os.path.exists(dest):  # resume: count, then skip
            done = pq.ParquetFile(dest).metadata.num_rows
            if done != n:
                fail(f"{dest} has {done} rows but the input batch has {n}; rerun with --force")
            countries.update(pq.read_table(dest, columns=["country"]).column(0).to_pylist())
            row0 += n
            part += 1
            continue
        ids = batch.column("entity_id")
        id_num = numeric_ids(ids, prefix)
        names = batch.column("business_name").to_pylist()
        addrs = batch.column("business_address").to_pylist()
        ctry = [c.strip() for c in batch.column("country").to_pylist()]
        name_b, addr_b, addr_ip, flags = [], [], [], np.zeros(n, np.uint16)
        for i in range(n):
            nb, fn = basic_name(names[i])
            ab, ip, fa = basic_address(addrs[i])
            name_b.append(nb)
            addr_b.append(ab)
            addr_ip.append(ip)
            flags[i] = fn | fa
            if len(examples) < 5:
                examples.append({"raw_name": names[i], "name_basic": nb, "raw_addr": addrs[i], "addr_basic": ab})
            if (fn & F_INDIC_NAME) and len(indic_examples) < 5:
                indic_examples.append({"raw_name": names[i], "name_basic": nb})
        countries.update(ctry)
        for k, m in FLAG_NAMES.items():
            stats[k] += int(((flags & m) != 0).sum())
        stats["empty_name"] += sum(1 for x in name_b if not x)
        table = pa.table({
            "row": pa.array(np.arange(row0, row0 + n, dtype=np.int32)),
            "id_num": pa.array(id_num), "entity_id": ids, "country": pa.array(ctry),
            "name_basic": pa.array(name_b), "addr_basic": pa.array(addr_b),
            "addr_indic_parts": pa.array(addr_ip), "flags": pa.array(flags),
        }, schema=SCHEMA)
        w = AtomicParquetWriter(dest, SCHEMA)
        w.write_table(table)
        w.close()
        row0 += n
        part += 1
        rate = row0 / max(time.time() - t0, 1e-6)
        log(f"{paths.split}/{src}: {row0:,} rows ({rate:,.0f} rows/s)")

    # id -> row lookup arrays + uniqueness check
    parts = sorted(glob.glob(os.path.join(out_dir, f"{src}-part-*.parquet")))
    id_all = np.concatenate([pq.read_table(p, columns=["id_num"]).column(0).to_numpy() for p in parts])
    order = np.argsort(id_all, kind="stable")
    ids_sorted = id_all[order]
    dup = int((ids_sorted[1:] == ids_sorted[:-1]).sum())
    if dup:
        fail(f"{src}: {dup} duplicate entity ids")
    np.save(os.path.join(out_dir, f"{src}_ids_sorted.npy"), ids_sorted)
    np.save(os.path.join(out_dir, f"{src}_ids_order.npy"), order.astype(np.int32))
    return {"rows": int(row0), "parts": len(parts), "countries": dict(countries), "flags": dict(stats),
            "examples": examples, "indic_examples": indic_examples}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--chunk-rows", type=int, default=config.CHUNK_ROWS)
    args = ap.parse_args()
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    sig = {"stage": "01", "chunk_rows": args.chunk_rows,
           "raw": {s: file_stamp(paths.raw(s)) for s in SOURCES if os.path.exists(paths.raw(s))}}
    if begin_stage(paths.normalized, sig, args.force):
        return
    check_ram(0.3 + args.chunk_rows * 1500 / 2**30, "01 normalize", args.force)
    summary = {"split": args.split, "chunk_rows": args.chunk_rows}
    for src in SOURCES:
        s = normalize_source(paths, src, args.chunk_rows, args.force)
        summary[src] = s
        log(f"{src}: {s['rows']:,} rows, countries {s['countries']}, flags {s['flags']}")
        for ex in s["examples"][:3] + s["indic_examples"][:3]:
            log(f"  example {ex}", mem=False)
    if summary["s1"]["countries"].keys() != (summary["s2"]["countries"].keys() | summary["s3"]["countries"].keys()):
        log("NOTE: country label sets differ between sources (handled generically; nothing is filtered)")
    write_manifest(paths.normalized, summary)


if __name__ == "__main__":
    main()
