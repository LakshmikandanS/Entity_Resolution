"""Stage 08: out-of-fold probabilities for EVERY training candidate pair (not just the sampled
training rows), so threshold tuning sees the true candidate distribution, singletons included.
A pair in fold f is scored only by model f, which never saw any pair of that S1.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.features import FEATURE_COLUMNS  # noqa: E402
from utils.gpu import GB, check_ram, release_gpu  # noqa: E402
from utils.io import (AtomicParquetWriter, add_common_args, clear_stage, limit_threads, list_shards, log,  # noqa: E402
                      paths_from_args, read_manifest, stage_done, write_manifest)
from utils.model import batch_matrix, load_model, model_paths, predict  # noqa: E402

SCHEMA = pa.schema([("s1", pa.int32()), ("tgt", pa.int32()), ("p", pa.float32()), ("label", pa.int8()),
                    ("fold", pa.int8())])


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--batch-rows", type=int, default=config.PREDICT_BATCH_ROWS)
    args = ap.parse_args()
    args.split = "train"
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    out_dir = paths.oof
    if stage_done(out_dir) and not args.force:
        log(f"{out_dir} already finished (use --force to rebuild)")
        return
    if args.force:
        clear_stage(out_dir)
    mman = read_manifest(paths.models, "stage 07")
    n_folds = mman["n_folds"]
    check_ram(0.5 + args.batch_rows * len(FEATURE_COLUMNS) * 4 * 2 / GB, "08 OOF", args.force)
    models = [load_model(p, n_threads=args.n_threads) for p in model_paths(paths.models, n_folds)]
    hist = {0: np.zeros(20, np.int64), 1: np.zeros(20, np.int64)}
    total = 0
    t0 = time.time()
    cols = FEATURE_COLUMNS + ["s1", "tgt", "label", "fold"]
    for p in list_shards(paths.features):
        dest = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(dest):
            t = pq.read_table(dest, columns=["p", "label"])
            pr, y = t.column(0).to_numpy(), t.column(1).to_numpy()
        else:
            w = AtomicParquetWriter(dest, SCHEMA)
            prs, ys = [], []
            for batch in pq.ParquetFile(p).iter_batches(batch_size=args.batch_rows, columns=cols):
                X = batch_matrix(batch)
                fold = batch.column("fold").to_numpy()
                pb = np.full(len(fold), np.nan, np.float32)
                for f in range(n_folds):
                    m = fold == f
                    if m.any():
                        pb[m] = predict(models[f], X[m])
                if np.isnan(pb).any():
                    raise RuntimeError(f"{p}: rows with a fold outside 0..{n_folds - 1}")
                yb = batch.column("label").to_numpy()
                w.write_table(pa.table({"s1": batch.column("s1"), "tgt": batch.column("tgt"), "p": pa.array(pb),
                                        "label": batch.column("label"), "fold": batch.column("fold")}, schema=SCHEMA))
                prs.append(pb); ys.append(yb)
            w.close()
            pr, y = np.concatenate(prs), np.concatenate(ys)
        for lab in (0, 1):
            hist[lab] += np.histogram(pr[y == lab], bins=20, range=(0, 1))[0]
        total += len(pr)
        log(f"OOF {os.path.basename(p)}: {total:,} predictions ({total / max(time.time() - t0, 1e-6):,.0f}/s)")
    release_gpu()
    stats = {"predictions": total, "prob_hist_negatives": hist[0].tolist(), "prob_hist_positives": hist[1].tolist(),
             "bins": np.linspace(0, 1, 21).tolist(), "backend": mman["backend"]}
    log(f"OOF probability histogram (20 bins) negatives {hist[0].tolist()}", mem=False)
    log(f"OOF probability histogram (20 bins) positives {hist[1].tolist()}", mem=False)
    write_manifest(out_dir, stats)


if __name__ == "__main__":
    main()
