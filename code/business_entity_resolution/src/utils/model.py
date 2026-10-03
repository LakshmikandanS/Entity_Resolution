"""Pair classifier: GPU XGBoost (preferred) or sklearn HistGradientBoosting (CPU fallback).

XGBoost never sees a materialised float matrix: an `xgboost.DataIter` streams parquet shards in
`TRAIN_ITER_ROWS` batches into a `QuantileDMatrix` (<= 1 byte per value after quantisation).
"""
import inspect
import os
import time

import numpy as np
import pyarrow.parquet as pq

from .features import FEATURE_COLUMNS
from .gpu import xgboost_status
from .io import log

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],   # monitoring only; the decision metric is macro F0.5
    "tree_method": "hist",
    "max_bin": 256,
    "max_depth": 8,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
}
XGB_ROUNDS = 1500
XGB_EARLY_STOP = 100

HGB_PARAMS = {
    "max_iter": 600, "learning_rate": 0.1, "max_leaf_nodes": 63, "max_bins": 255,
    "l2_regularization": 1.0, "min_samples_leaf": 50, "early_stopping": True,
    "n_iter_no_change": 30,
}


def choose_backend(requested="auto"):
    ok, ver, cuda = xgboost_status()
    if requested == "xgboost":
        if not ok:
            raise RuntimeError("xgboost requested but not installed")
        return "xgboost", ("cuda" if cuda else "cpu")
    if requested == "hgb":
        return "hgb", "cpu"
    if ok:
        return "xgboost", ("cuda" if cuda else "cpu")
    return "hgb", "cpu"


def _entity_keep(s1, keep_frac):
    """Deterministic entity-level subsample: keeps every row of an S1 or none of them."""
    s1 = s1.astype(np.uint64)
    return ((s1 * np.uint64(2654435761)) % np.uint64(1_000_003)) < np.uint64(int(keep_frac * 1_000_003))


def _row_filter(batch, folds_excluded, val_mode, keep_frac=1.0):
    fold = batch.column("fold").to_numpy()
    es_val = batch.column("es_val").to_numpy(zero_copy_only=False).astype(bool)
    m = ~np.isin(fold, folds_excluded) if folds_excluded is not None else np.ones(len(fold), bool)
    if val_mode == "train":
        m &= ~es_val
    elif val_mode == "val":
        m &= es_val
    if keep_frac < 1.0:
        m &= _entity_keep(batch.column("s1").to_numpy(), keep_frac)
    return m


def _xy(batch, mask, features):
    X = np.empty((int(mask.sum()), len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = batch.column(name).to_numpy(zero_copy_only=False)[mask]
    y = batch.column("label").to_numpy(zero_copy_only=False)[mask].astype(np.float32)
    w = batch.column("weight").to_numpy(zero_copy_only=False)[mask].astype(np.float32)
    return X, y, w


def make_iter(paths, folds_excluded, val_mode, batch_rows, keep_frac=1.0, features=FEATURE_COLUMNS):
    import xgboost as xgb
    cols = list(features) + ["label", "weight", "fold", "es_val", "s1"]

    class ShardIter(xgb.DataIter):
        def __init__(self):
            self._gen = None
            self.rows = 0
            super().__init__()

        def _batches(self):
            for p in paths:
                for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_rows, columns=cols):
                    m = _row_filter(batch, folds_excluded, val_mode, keep_frac)
                    if m.any():
                        yield _xy(batch, m, features)

        def reset(self):
            self._gen = self._batches()
            self.rows = 0

        def next(self, input_data):
            if self._gen is None:
                self.reset()
            try:
                X, y, w = next(self._gen)
            except StopIteration:
                return False
            self.rows += len(y)
            input_data(data=X, label=y, weight=w)
            return True

    return ShardIter()


def train_xgboost(paths, fold, n_threads, device, batch_rows, keep_frac=1.0, params=None, rounds=XGB_ROUNDS):
    import xgboost as xgb
    params = dict(XGB_PARAMS, **(params or {}))
    params.update({"device": device, "nthread": n_threads, "seed": 2026 + fold})
    t0 = time.time()
    it_tr = make_iter(paths, [fold], "train", batch_rows, keep_frac)
    dtrain = xgb.QuantileDMatrix(it_tr, max_bin=params["max_bin"])
    it_va = make_iter(paths, [fold], "val", batch_rows, keep_frac)
    dval = xgb.QuantileDMatrix(it_va, max_bin=params["max_bin"], ref=dtrain)
    log(f"fold {fold}: QuantileDMatrix train {dtrain.num_row():,} x {dtrain.num_col()} "
        f"val {dval.num_row():,} built in {time.time() - t0:.0f}s")
    evals_result = {}
    booster = xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=XGB_EARLY_STOP, evals_result=evals_result, verbose_eval=50)
    info = {"backend": "xgboost", "device": device, "params": params,
            "best_iteration": int(booster.best_iteration), "train_rows": int(dtrain.num_row()),
            "val_rows": int(dval.num_row()), "keep_frac": keep_frac, "seconds": time.time() - t0,
            "val_logloss_best": float(min(evals_result["val"]["logloss"]))}
    del dtrain, dval
    return booster, info


def load_rows_in_memory(paths, folds_excluded, val_mode, max_rows, features=FEATURE_COLUMNS):
    """For the CPU fallback: read the filtered rows, subsampling whole S1 entities if needed."""
    cols = list(features) + ["label", "weight", "fold", "es_val", "s1"]
    total = 0
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1_000_000, columns=["fold", "es_val"]):
            total += int(_row_filter(batch, folds_excluded, val_mode).sum())
    keep_frac = min(1.0, max_rows / max(total, 1))
    Xs, ys, ws = [], [], []
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1_000_000, columns=cols):
            m = _row_filter(batch, folds_excluded, val_mode, keep_frac)
            if m.any():
                X, y, w = _xy(batch, m, features)
                Xs.append(X); ys.append(y); ws.append(w)
    if not Xs:
        return np.zeros((0, len(features)), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), total
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), total


def train_hgb(paths, fold, max_rows):
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    X, y, w, total = load_rows_in_memory(paths, [fold], "train", max_rows)
    Xv, yv, wv, _ = load_rows_in_memory(paths, [fold], "val", max(1, max_rows // 10))
    log(f"fold {fold}: HGB rows {len(y):,} of {total:,} available, val {len(yv):,}")
    clf = HistGradientBoostingClassifier(random_state=2026 + fold, **HGB_PARAMS)
    fit_kwargs = {"sample_weight": w}
    if "X_val" in inspect.signature(clf.fit).parameters and len(yv):
        fit_kwargs.update({"X_val": Xv, "y_val": yv, "sample_weight_val": wv})
    else:
        clf.set_params(validation_fraction=0.1)
    clf.fit(X, y, **fit_kwargs)
    info = {"backend": "hgb", "device": "cpu", "params": HGB_PARAMS, "n_iter": int(clf.n_iter_),
            "train_rows": int(len(y)), "available_rows": int(total), "seconds": time.time() - t0}
    return clf, info


def save_model(model, backend, path_noext):
    if backend == "xgboost":
        path = path_noext + ".ubj"
        model.save_model(path)
    else:
        import joblib
        path = path_noext + ".joblib"
        joblib.dump(model, path)
    return path


def load_model(path, device="cuda", n_threads=4):
    if path.endswith(".ubj"):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(path)
        _, _, cuda = xgboost_status()
        booster.set_param({"device": device if cuda else "cpu", "nthread": n_threads})
        return ("xgboost", booster)
    import joblib
    return ("hgb", joblib.load(path))


def predict(model, X):
    backend, m = model
    if backend == "xgboost":
        best = getattr(m, "best_iteration", None)
        rng = (0, int(best) + 1) if best is not None else (0, 0)
        return m.inplace_predict(X, iteration_range=rng).astype(np.float32)
    return m.predict_proba(X)[:, 1].astype(np.float32)


def batch_matrix(batch, features=FEATURE_COLUMNS):
    X = np.empty((batch.num_rows, len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = batch.column(name).to_numpy(zero_copy_only=False)
    return X


def model_paths(models_dir, n_folds):
    out = []
    for f in range(n_folds):
        for ext in (".ubj", ".joblib"):
            p = os.path.join(models_dir, f"fold{f}{ext}")
            if os.path.exists(p):
                out.append(p)
                break
        else:
            raise FileNotFoundError(f"no model for fold {f} in {models_dir}")
    return out
