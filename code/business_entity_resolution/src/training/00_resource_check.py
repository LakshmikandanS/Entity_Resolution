"""Stage 00: report RAM / GPU / library status and the per-stage resource plan for this machine.

Reads only file metadata (sizes, line counts are estimated from bytes). With --probe-xgboost it trains
a 2,000-row synthetic model on CUDA to prove the installed xgboost build supports this GPU.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config  # noqa: E402
from utils.features import FEATURE_COLUMNS, estimate_batch_vram_bytes  # noqa: E402
from utils.gpu import GB, gpu_report, library_versions, ram_report, xgboost_status  # noqa: E402
from utils.io import Paths, add_common_args, paths_from_args, write_json  # noqa: E402
from utils.records import REC_DTYPE  # noqa: E402

BYTES_PER_ROW = {"s1": 95.2, "s2": 97.2, "s3": 95.3}   # measured on train files (bytes / rows)


def est_rows(path, src):
    return int(os.path.getsize(path) / BYTES_PER_ROW[src]) if os.path.exists(path) else 0


def plan(paths, split, k):
    n1 = est_rows(paths.raw("s1"), "s1")
    nt = est_rows(paths.raw("s2"), "s2") + est_rows(paths.raw("s3"), "s3")
    pairs = n1 * k
    f = len(FEATURE_COLUMNS)
    rec = REC_DTYPE.itemsize
    keys = nt * 13
    return {
        "split": split, "s1_rows_est": n1, "target_rows_est": nt, "pairs_upper": pairs,
        "records_disk_gb": (n1 + nt) * rec / GB,
        "index_temp_keys_gb": keys * 13 / GB,
        "index_build_peak_gb": keys / config.INDEX_PARTITIONS * 32 / GB,
        "candidate_chunk_peak_gb": config.MAX_EXPANDED_POSTINGS * 50 / GB,
        "candidates_disk_gb": pairs * 14 / GB,
        "features_raw_gb": pairs * f * 4 / GB,
        "feature_batch_vram_gb": estimate_batch_vram_bytes(config.BATCH_SIZE) / GB,
        "target_competition_arrays_gb": nt * 24 / GB,
        "predict_batch_gb": config.PREDICT_BATCH_ROWS * f * 4 / GB,
        "train_rows_est": int(n1 * 9.5) if split == "train" else 0,
        "xgb_vram_per_fold_gb": n1 * 9.5 / max(config.N_FOLDS, 1) * (config.N_FOLDS - 1) * (f + 24) / GB
        if split == "train" else 0,
    }


def probe_xgboost():
    import numpy as np
    import xgboost as xgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 10)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=2000) > 0).astype(np.float32)
    bst = xgb.train({"tree_method": "hist", "device": "cuda", "max_depth": 3}, xgb.QuantileDMatrix(X, y),
                    num_boost_round=5)
    return float(bst.inplace_predict(X[:5]).mean())


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--probe-xgboost", action="store_true")
    args = ap.parse_args()
    out = {"config": config.as_dict(), "ram": ram_report(), "gpu": gpu_report(),
           "libraries": library_versions()}
    ok, ver, cuda = xgboost_status()
    out["xgboost"] = {"installed": ok, "version": ver, "built_with_cuda": cuda}
    if args.probe_xgboost and ok and cuda:
        try:
            out["xgboost"]["cuda_probe_mean_pred"] = probe_xgboost()
            out["xgboost"]["cuda_probe"] = "ok"
        except Exception as e:  # e.g. wheel without sm_120 kernels
            out["xgboost"]["cuda_probe"] = f"FAILED: {e}"
    out["disk_free_gb"] = shutil.disk_usage(os.path.abspath(args.work_dir if os.path.exists(args.work_dir)
                                                             else args.data_dir)).free / GB
    for split in ("train", "test"):
        p = Paths(args.data_dir, args.work_dir, args.artifacts_dir, split)
        out[f"plan_{split}"] = plan(p, split, config.MAX_CANDIDATES_PER_S1)

    r, g = out["ram"], out["gpu"]
    print(f"RAM   total {r['total_gb']:.1f} GB, available {r['available_gb']:.1f} GB "
          f"(budget RAM_BUDGET_GB={config.RAM_BUDGET_GB})")
    print(f"GPU   {g.get('name')}  total {g.get('total_gb', 0):.1f} GB  free {g.get('free_gb', 0):.1f} GB  "
          f"CUDA available={g['cuda_available']} (torch {g.get('torch')}, CUDA {g.get('cuda_version')})")
    print(f"XGB   installed={ok} version={ver} cuda={cuda} probe={out['xgboost'].get('cuda_probe', 'not run')}")
    if not ok:
        print("      -> the pipeline will use sklearn HistGradientBoosting on CPU (row-capped).")
    print("LIBS  " + ", ".join(f"{k}={v}" for k, v in out["libraries"].items()))
    print(f"DISK  free {out['disk_free_gb']:.0f} GB")
    for split in ("train", "test"):
        print(f"\nPlan [{split}]")
        for k, v in out[f"plan_{split}"].items():
            print(f"  {k:32s} {v:,.2f}" if isinstance(v, float) else f"  {k:32s} {v:,}" if isinstance(v, int) else f"  {k:32s} {v}")
    if r["available_gb"] < 3.0:
        print("\nWARNING: less than 3 GB RAM available; close other programs before stages 03-05.")
    write_json(paths_from_args(args).artifact("resource_check.json"), out)


if __name__ == "__main__":
    main()
