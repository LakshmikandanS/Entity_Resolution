"""Stage 07: entity-level cross-fitted pair classifiers (problem.md decision D5).

fold(S1) = crc32(S1 entity_id) % 1000 % N_FOLDS, so all candidate pairs of an S1 share a fold.
Model f is trained on training rows of the other folds (es_val rows of those folds are used only
for early stopping) and later predicts fold f out-of-fold (08) and the test set (11).

Backend: GPU XGBoost (QuantileDMatrix fed by a streaming DataIter) when xgboost is installed with
CUDA; otherwise sklearn HistGradientBoosting on CPU with an entity-level row cap.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config  # noqa: E402
from utils.features import FEATURE_COLUMNS  # noqa: E402
from utils.gpu import GB, check_ram, gpu_report, ram_report, release_gpu  # noqa: E402
from utils.io import (add_common_args, begin_stage, list_shards, log, paths_from_args, read_json, read_manifest, upstream_stamp,  # noqa: E402
                      write_json, write_manifest)
from utils.model import choose_backend, save_model, train_hgb, train_xgboost  # noqa: E402


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--backend", choices=["auto", "xgboost", "hgb"], default="auto")
    ap.add_argument("--max-gpu-gb", type=float, default=config.MAX_GPU_MEMORY_GB)
    ap.add_argument("--max-train-rows", type=int, default=3_000_000, help="CPU (HGB) fallback row cap")
    ap.add_argument("--iter-rows", type=int, default=config.TRAIN_ITER_ROWS)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="train only these folds")
    args = ap.parse_args()
    args.split = "train"
    paths = paths_from_args(args)
    tman = read_manifest(paths.trainset, "stage 06")
    n_folds = read_manifest(paths.labels, "stage 02")["n_folds"]
    shards = list_shards(paths.trainset)
    backend, device = choose_backend(args.backend)
    sig = {"stage": "07", "trainset": upstream_stamp(paths.trainset), "backend": backend, "device": device,
           "max_train_rows": args.max_train_rows if backend == "hgb" else None}
    if args.folds is None and begin_stage(paths.models, sig, args.force):
        return
    log(f"backend {backend} on {device}; {tman['rows']:,} training rows in {len(shards)} shards")
    if backend == "hgb":
        log("NOTE: xgboost with CUDA not available -> CPU HistGradientBoosting with "
            f"--max-train-rows {args.max_train_rows:,} (entity-level subsample)")

    rows_per_fold = tman["rows"] * (n_folds - 1) / n_folds
    keep_frac = 1.0
    if backend == "xgboost" and device == "cuda":
        est = rows_per_fold * (len(FEATURE_COLUMNS) * 1 + 24) / GB + 0.3
        log(f"estimated VRAM per fold: {est:.2f} GB (cap {args.max_gpu_gb} GB)")
        if est > args.max_gpu_gb * 0.8:
            keep_frac = args.max_gpu_gb * 0.8 / est
            log(f"  -> subsampling whole S1 entities to {keep_frac:.2%} of rows to stay within budget")
        check_ram(0.5 + rows_per_fold * keep_frac * len(FEATURE_COLUMNS) / GB + args.iter_rows * 70 * 4 * 2 / GB,
                  "07 train (xgboost)", args.force)
    elif backend == "xgboost":
        check_ram(0.5 + rows_per_fold * len(FEATURE_COLUMNS) * 1.2 / GB, "07 train (xgboost cpu)", args.force)
    else:
        check_ram(0.5 + args.max_train_rows * len(FEATURE_COLUMNS) * 12 / GB, "07 train (hgb)", args.force)

    infos = {}
    for fold in (args.folds if args.folds is not None else range(n_folds)):
        base = os.path.join(paths.models, f"fold{fold}")
        if any(os.path.exists(base + e) for e in (".ubj", ".joblib")) and not args.force:
            log(f"fold {fold}: model exists, skipping (use --force to retrain)")
            continue
        t0 = time.time()
        if backend == "xgboost":
            model, info = train_xgboost(shards, fold, args.n_threads, device, args.iter_rows, keep_frac)
        else:
            model, info = train_hgb(shards, fold, args.max_train_rows)
        info.update({"fold": fold, "features": FEATURE_COLUMNS, "ram_after": ram_report(), "gpu_after": gpu_report(),
                     "wall_seconds": time.time() - t0})
        path = save_model(model, backend, base)
        write_json(base + ".json", info)
        infos[fold] = info
        log(f"fold {fold}: saved {path} ({info['wall_seconds']:.0f}s)")
        del model
        release_gpu()
    for fold in range(n_folds):   # include folds trained in earlier runs
        meta = os.path.join(paths.models, f"fold{fold}.json")
        if fold not in infos and os.path.exists(meta):
            infos[fold] = read_json(meta)
    missing = [f for f in range(n_folds) if f not in infos]
    if missing:
        log(f"folds still missing: {missing}; not writing the stage manifest yet")
        return
    write_manifest(paths.models, {"backend": backend, "device": device, "n_folds": n_folds,
                                  "keep_frac": keep_frac, "folds": infos, "features": FEATURE_COLUMNS})


if __name__ == "__main__":
    main()
