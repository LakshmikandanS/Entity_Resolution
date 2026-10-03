"""Stage 11: test inference -> output/matching_results.tsv and output/candidate_pairs.tsv.

Prerequisite (same code, --split test): 01_normalize, 03_build_blocking_indexes, 04_generate_candidates,
05_build_features. Nothing here reads labels or tunes anything: tau, the models and the rewrite map
all come from artifacts/final/ (training data only).

Decision (identical to 09): p = mean of the fold models; each target goes to its highest-p S1 only;
keep it if p >= tau. candidate_pairs.tsv is exactly the set of pairs the model scored.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.decision import Exclusivity  # noqa: E402
from utils.features import FEATURE_COLUMNS  # noqa: E402
from utils.gpu import GB, check_ram, release_gpu  # noqa: E402
from utils.io import (AtomicParquetWriter, add_common_args, fail, limit_threads, list_shards, log,  # noqa: E402
                      paths_from_args, read_json, read_manifest, write_json)
from utils.model import batch_matrix, load_model, predict  # noqa: E402

PRED_SCHEMA = pa.schema([("s1", pa.int32()), ("tgt", pa.int32()), ("p", pa.float32())])


def entity_ids(norm_dir, src):
    parts = sorted(glob.glob(os.path.join(norm_dir, f"{src}-part-*.parquet")))
    return pa.concat_arrays([c for p in parts for c in pq.read_table(p, columns=["entity_id"]).column(0).chunks])


def grouped_lines(s1_ids, lo, hi, s1_rows, tgt_ids_str):
    """One line per S1 row in [lo, hi): '<s1 id>\\t<comma-joined ids>' (empty list allowed)."""
    lines = []
    starts = np.searchsorted(s1_rows, np.arange(lo, hi + 1))
    for r in range(lo, hi):
        a, b = starts[r - lo], starts[r - lo + 1]
        lines.append(f"{s1_ids[r]}\t{','.join(tgt_ids_str[a:b])}")
    return lines


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--model-set", choices=["folds", "full"], default="folds")
    ap.add_argument("--batch-rows", type=int, default=config.PREDICT_BATCH_ROWS)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="override tau (not recommended)")
    args = ap.parse_args()
    args.split = "test"
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    out_dir = args.output_dir or os.path.join(os.path.dirname(paths.work), "output")
    os.makedirs(out_dir, exist_ok=True)
    final = paths.artifact("final")
    bundle = read_json(os.path.join(final, "bundle.json"))
    if bundle["features"] != FEATURE_COLUMNS:
        fail("feature list of the bundle differs from the code; retrain or check out the matching code")
    tau = args.threshold if args.threshold is not None else bundle["threshold"]
    nman = read_manifest(paths.normalized, "stage 01 (test)")
    iman = read_manifest(paths.index, "stage 03 (test)")
    cman = read_manifest(paths.candidates, "stage 04 (test)")
    read_manifest(paths.features, "stage 05 (test)")
    n1, n2, n_t = iman["n_s1"], iman["n2"], iman["n_targets"]
    check_ram(0.6 + args.batch_rows * len(FEATURE_COLUMNS) * 4 * 2 / GB + n_t * 8 / GB + n_t * 20 / GB,
              "11 predict", args.force)

    # ---- predict
    if args.model_set == "folds":
        mpaths = [os.path.join(final, f) for f in sorted(bundle["files"]) if f.startswith("fold")]
    else:
        mpaths = [os.path.join(final, f) for f in bundle["files"] if f.startswith("full")]
    if not mpaths:
        fail(f"no '{args.model_set}' models in {final}")
    models = [load_model(p, n_threads=args.n_threads) for p in mpaths]
    log(f"models: {[os.path.basename(p) for p in mpaths]}, tau = {tau:.6f}")
    t0 = time.time()
    total = 0
    for p in list_shards(paths.features):
        dest = os.path.join(paths.predictions, os.path.basename(p))
        if os.path.exists(dest):
            total += pq.ParquetFile(dest).metadata.num_rows
            continue
        w = AtomicParquetWriter(dest, PRED_SCHEMA)
        for batch in pq.ParquetFile(p).iter_batches(batch_size=args.batch_rows, columns=FEATURE_COLUMNS + ["s1", "tgt"]):
            X = batch_matrix(batch)
            pr = np.mean([predict(m, X) for m in models], axis=0).astype(np.float32)
            w.write_table(pa.table({"s1": batch.column("s1"), "tgt": batch.column("tgt"), "p": pa.array(pr)},
                                   schema=PRED_SCHEMA))
            total += len(pr)
        w.close()
        log(f"predicted {total:,} pairs ({total / max(time.time() - t0, 1e-6):,.0f}/s)")
    release_gpu()

    # ---- exclusivity + threshold
    pshards = list_shards(paths.predictions)
    ex = Exclusivity(n_t)
    for step in (1, 2):
        for p in pshards:
            t = pq.read_table(p)
            s1, tgt, pr = (t.column(c).to_numpy() for c in ("s1", "tgt", "p"))
            ex.pass1(tgt, pr) if step == 1 else ex.pass1b(s1, tgt, pr)

    # ---- write outputs, shard by shard (shards are contiguous S1 ranges)
    s1_ids = entity_ids(paths.normalized, "s1").to_pylist()
    if len(s1_ids) != n1:
        fail("S1 id count mismatch")
    tgt_ids = pa.concat_arrays([entity_ids(paths.normalized, "s2"), entity_ids(paths.normalized, "s3")])
    per = cman["s1_per_shard"]
    m_path, c_path = os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
    assigned_all, n_links, n_cands, s1_with = [], 0, 0, 0
    with open(m_path + ".tmp", "w", encoding="utf-8", newline="\n") as fm, \
            open(c_path + ".tmp", "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for k, p in enumerate(pshards):
            lo, hi = k * per, min(n1, (k + 1) * per)
            t = pq.read_table(p)
            s1, tgt, pr = (t.column(c).to_numpy() for c in ("s1", "tgt", "p"))
            if len(s1) and (s1.min() < lo or s1.max() >= hi or np.any(np.diff(s1) < 0)):
                fail(f"{p}: S1 rows outside [{lo}, {hi}) or not sorted")
            ids = tgt_ids.take(pa.array(tgt)).to_pylist()
            fc.write("\n".join(grouped_lines(s1_ids, lo, hi, s1, ids)) + "\n")
            keep = ex.winners(s1, tgt, pr) & (pr >= tau)
            ks1 = s1[keep]
            kids = [ids[i] for i in np.flatnonzero(keep)]
            fm.write("\n".join(grouped_lines(s1_ids, lo, hi, ks1, kids)) + "\n")
            assigned_all.append(tgt[keep])
            n_links += int(keep.sum())
            n_cands += len(s1)
            s1_with += len(np.unique(ks1))
        if (len(pshards) * per) < n1:
            fail("candidate shards do not cover every S1 row")
    assigned = np.concatenate(assigned_all) if assigned_all else np.zeros(0, np.int32)
    ex.check(assigned)
    os.replace(m_path + ".tmp", m_path)
    os.replace(c_path + ".tmp", c_path)

    report = {"s1": n1, "candidate_pairs": n_cands, "predicted_links": n_links, "s1_with_matches": s1_with,
              "s1_empty": n1 - s1_with, "empty_rate": (n1 - s1_with) / n1, "threshold": tau,
              "duplicate_target_assignments": 0, "model_set": args.model_set,
              "links_from_s2": int((assigned < n2).sum()), "links_from_s3": int((assigned >= n2).sum()),
              "normalized_countries_s1": nman["s1"]["countries"]}
    log(f"wrote {m_path}: {n_links:,} links, {s1_with:,}/{n1:,} S1 with matches "
        f"(empty {report['empty_rate']:.2%}; training singleton rate was ~5.6%)")
    log(f"wrote {c_path}: {n_cands:,} candidate pairs")
    write_json(paths.artifact("test_prediction_report.json"), report)

    validator = os.path.join(os.path.dirname(paths.data_dir), "utils", "validate_submission.py")
    if os.path.exists(validator):
        rc = subprocess.run([sys.executable, validator, "--matching", m_path, "--candidate", c_path,
                             "--test-dir", os.path.join(paths.data_dir, "test")]).returncode
        log(f"validate_submission.py exit code {rc}")
        if rc != 0:
            sys.exit(rc)
    else:
        log(f"NOTE: {validator} not found; run the official validator before submitting")


if __name__ == "__main__":
    main()
