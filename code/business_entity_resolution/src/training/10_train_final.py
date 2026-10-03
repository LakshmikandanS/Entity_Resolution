"""Stage 10: assemble the final, self-describing model bundle in artifacts/final/.

Default (recommended): the N_FOLDS cross-fitted models; test probability = their mean (D5). This
keeps the scores on the same footing as the OOF scores tau was tuned on.
--refit-full additionally trains one model on all training entities (es_val rows still used for
early stopping) and stores it as full.*; 11_predict_test.py uses it only with --model-set full.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config  # noqa: E402
from utils.features import FEATURE_COLUMNS  # noqa: E402
from utils.gpu import library_versions, release_gpu  # noqa: E402
from utils.io import (add_common_args, fail, list_shards, log, paths_from_args, read_json, read_manifest,  # noqa: E402
                      write_json, write_manifest)
from utils.model import choose_backend, model_paths, save_model, train_hgb, train_xgboost  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--refit-full", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=3_000_000)
    args = ap.parse_args()
    args.split = "train"
    paths = paths_from_args(args)
    mman = read_manifest(paths.models, "stage 07")
    thr_path = paths.artifact("threshold.json")
    if not os.path.exists(thr_path):
        fail("artifacts/threshold.json missing: run 09_tune_threshold.py first")
    final = paths.artifact("final")
    os.makedirs(final, exist_ok=True)
    files = {}
    for p in model_paths(paths.models, mman["n_folds"]):
        dst = os.path.join(final, os.path.basename(p))
        shutil.copy2(p, dst)
        files[os.path.basename(p)] = sha256(dst)
    if args.refit_full:
        backend, device = choose_backend(mman["backend"])
        shards = list_shards(paths.trainset)
        t0 = time.time()
        if backend == "xgboost":
            model, info = train_xgboost(shards, -1, args.n_threads, device, config.TRAIN_ITER_ROWS,
                                        mman.get("keep_frac", 1.0))
        else:
            model, info = train_hgb(shards, -1, args.max_train_rows)
        p = save_model(model, backend, os.path.join(final, "full"))
        files[os.path.basename(p)] = sha256(p)
        write_json(os.path.join(final, "full.json"), info)
        log(f"full refit done in {time.time() - t0:.0f}s")
        del model
        release_gpu()
    for name in ("threshold.json", "rewrite_map.json"):
        shutil.copy2(paths.artifact(name), os.path.join(final, name))
        files[name] = sha256(os.path.join(final, name))
    bundle = {"backend": mman["backend"], "n_folds": mman["n_folds"], "features": FEATURE_COLUMNS,
              "threshold": read_json(thr_path)["threshold"], "files": files, "config": config.as_dict(),
              "libraries": library_versions(), "default_model_set": "folds"}
    write_json(os.path.join(final, "bundle.json"), bundle)
    write_manifest(final, {"files": list(files)})
    log(f"final bundle in {final}: {list(files)}; tau = {bundle['threshold']:.6f}")


if __name__ == "__main__":
    main()
