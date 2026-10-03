"""Adapter between the neural experiment and the classical baseline in code/business_entity_resolution/src.

The neural branch never re-implements baseline logic. Normalisation, rewrite map, blocking, candidate
pairs, the 64 handcrafted features, training-row construction, folds, the XGBoost/HGB model, target
exclusivity, the exact macro-F0.5 sweep and the output writer all come from the baseline's own modules
and artefacts. Baseline files are imported, never modified.

What the neural scripts read from the baseline (all produced by its stages, see its README):
  work/<split>/normalized/_MANIFEST.json     row counts per source (01)
  artifacts/rewrite_map.json                 learned rewrite map, train only (02)
  work/train/labels/{n_true,s1_bucket}.npy   labels and entity-level folds (02)
  work/<split>/candidates/_MANIFEST.json     s1_per_shard (04)
  work/<split>/features/part-*.parquet       candidate pairs + 64 features (+ label/fold/es_val) (05)
  work/train/trainset/part-*.parquet         sampled training rows with weights (06)

Conventions shared with the baseline
  pair_idx  : position of a pair in the concatenation of work/<split>/features shards (sorted names).
              Rows are sorted by (s1, blk_rank), so all pairs of one S1 are contiguous.
  s1_row    : baseline `s1` = data-row index in <split>_source1.tsv.
  tgt / gid : baseline `tgt` = S2 rows first, then S3 rows (same convention as n04's t_gid).
"""
from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class BaselineNotReady(RuntimeError):
    pass


class ClassicalBaseline:
    def __init__(self, cfg):
        bc = cfg["baseline"]
        self.src = (cfg.root / bc["src"]).resolve()
        if not (self.src / "utils" / "features.py").exists():
            raise BaselineNotReady(f"baseline source not found at {self.src}")
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
        self.io = importlib.import_module("utils.io")
        self.features = importlib.import_module("utils.features")
        self.model = importlib.import_module("utils.model")
        self.metrics = importlib.import_module("utils.metrics")
        self.decision = importlib.import_module("utils.decision")
        self.norm = importlib.import_module("utils.normalization")
        self.config = importlib.import_module("utils.config")
        data = cfg.root / bc["data_dir"] if bc.get("data_dir") else cfg.dataset
        self._dirs = (str(data), str(cfg.root / bc["work_dir"]), str(cfg.root / bc["artifacts_dir"]))
        self._canon = None

    # ------------------------------------------------------------------ paths / manifests
    def paths(self, split):
        return self.io.Paths(*self._dirs, split=split)

    def _manifest(self, directory, what):
        if not self.io.stage_done(directory):
            raise BaselineNotReady(f"baseline {what} is not finished ({directory} has no _MANIFEST.json)")
        return self.io.read_json(os.path.join(directory, self.io.MANIFEST))

    def sizes(self, split):
        m = self._manifest(self.paths(split).normalized, f"stage 01 ({split})")
        return m["s1"]["rows"], m["s2"]["rows"], m["s3"]["rows"]

    def feature_shards(self, split):
        self._manifest(self.paths(split).features, f"stage 05 ({split})")
        return self.io.list_shards(self.paths(split).features)

    def shard_offsets(self, split):
        shards = self.feature_shards(split)
        rows = [pq.ParquetFile(p).metadata.num_rows for p in shards]
        return shards, np.r_[0, np.cumsum(rows)].astype(np.int64)

    def num_pairs(self, split):
        return int(self._manifest(self.paths(split).features, f"stage 05 ({split})")["rows"])

    def s1_per_shard(self, split):
        return int(self._manifest(self.paths(split).candidates, f"stage 04 ({split})")["s1_per_shard"])

    # ------------------------------------------------------------------ pairs
    def iter_pairs(self, split, columns, batch_rows):
        """dicts of numpy arrays in pair_idx order; columns from {s1_row, t_src, t_row, cand_rank, label}."""
        n2 = self.sizes(split)[1]
        src_cols = {"s1_row": "s1", "t_src": "tgt", "t_row": "tgt", "cand_rank": "blk_rank", "label": "label"}
        read = sorted({src_cols[c] for c in columns})
        for _, b in self.io.iter_parquet(self.feature_shards(split), columns=read, batch_rows=batch_rows):
            raw = {c: b.column(c).to_numpy() for c in read}
            out = {}
            if "s1_row" in columns:
                out["s1_row"] = raw["s1"].astype(np.int32)
            if "t_src" in columns or "t_row" in columns:
                tg = raw["tgt"].astype(np.int64)
                out["t_src"] = np.where(tg < n2, 2, 3).astype(np.int8)
                out["t_row"] = np.where(tg < n2, tg, tg - n2).astype(np.int32)
            if "cand_rank" in columns:
                out["cand_rank"] = raw["blk_rank"].astype(np.int16) + 1      # baseline rank is 0-based
            if "label" in columns:
                out["label"] = raw["label"].astype(np.int8)
            yield out

    # ------------------------------------------------------------------ normalisation keys
    def _canonicalizer(self):
        if self._canon is None:
            path = self.paths("train").artifact("rewrite_map.json")
            if not os.path.exists(path):
                raise BaselineNotReady(f"{path} missing; run baseline stage 02 first")
            self._canon = self.norm.Canonicalizer(self.io.read_json(path))
        return self._canon

    def name_key(self, names, countries):
        """Baseline core name (transliteration + learned rewrite map + legal forms removed)."""
        c = self._canonicalizer()
        out = []
        for n in names:
            nb, fl = self.norm.basic_name(n)
            out.append(" ".join(c.canon_name(nb, fl)[1]))
        return out

    def address_key(self, addresses, countries):
        """Baseline order-insensitive canonical address ('' when empty)."""
        c = self._canonicalizer()
        out = []
        for a, ctry in zip(addresses, countries):
            ab = self.norm.basic_address(a)[0]
            out.append(c.canon_address(ab, ctry)["sorted_str"] if ab else "")
        return out

    # ------------------------------------------------------------------ labels / folds
    def n_folds(self):
        return int(self._manifest(self.paths("train").labels, "stage 02")["n_folds"])

    def s1_folds(self):
        bucket = np.load(os.path.join(self.paths("train").labels, "s1_bucket.npy")).astype(np.int64)
        return (bucket % self.n_folds()).astype(np.int8)

    def n_true(self):
        return np.load(os.path.join(self.paths("train").labels, "n_true.npy")).astype(np.int64)

    # ------------------------------------------------------------------ features / model
    def feature_names(self):
        return list(self.features.FEATURE_COLUMNS)

    def batch_matrix(self, batch, features=None):
        return self.model.batch_matrix(batch, features or self.features.FEATURE_COLUMNS)

    def trainset_shards(self):
        self._manifest(self.paths("train").trainset, "stage 06")
        return self.io.list_shards(self.paths("train").trainset)

    def choose_backend(self, requested="auto"):
        return self.model.choose_backend(requested)

    def xgb_keep_frac(self, rows_per_fold, n_features, max_gpu_gb):
        """Same VRAM budget rule as baseline training/07_train.py (entity subsample when over budget)."""
        gb = importlib.import_module("utils.gpu").GB
        est = rows_per_fold * (n_features * 1 + 24) / gb + 0.3
        return max_gpu_gb * 0.8 / est if est > max_gpu_gb * 0.8 else 1.0

    def fit_fold(self, shards, fold, features, backend, device, n_threads, iter_rows, keep_frac, max_train_rows):
        """Baseline train_xgboost / train_hgb on `shards`, with the feature list injected.

        The baseline functions read FEATURE_COLUMNS through the default argument of make_iter /
        load_rows_in_memory; both are module globals looked up at call time, so they are swapped for
        partials carrying `features` for the duration of the call and restored afterwards."""
        m = self.model
        orig_iter, orig_load = m.make_iter, m.load_rows_in_memory
        m.make_iter = functools.partial(orig_iter, features=features)
        m.load_rows_in_memory = functools.partial(orig_load, features=features)
        try:
            if backend == "xgboost":
                return m.train_xgboost(shards, fold, n_threads, device, iter_rows, keep_frac)
            return m.train_hgb(shards, fold, max_train_rows)
        finally:
            m.make_iter, m.load_rows_in_memory = orig_iter, orig_load

    def save_model(self, model, backend, path_noext):
        return self.model.save_model(model, backend, str(path_noext))

    def load_model(self, path, n_threads=4):
        return self.model.load_model(str(path), n_threads=n_threads)

    def predict(self, model, X):
        return self.model.predict(model, X)

    # ------------------------------------------------------------------ decision + metric
    def exclusivity_winners(self, n_targets, chunks):
        """chunks: callable returning an iterator of (s1, tgt, p) numpy arrays (scored pairs only).
        Returns a function mask(s1, tgt, p) -> winners, using baseline utils.decision.Exclusivity."""
        ex = self.decision.Exclusivity(n_targets)
        for s1, tgt, p in chunks():
            ex.pass1(tgt, p)
        for s1, tgt, p in chunks():
            ex.pass1b(s1, tgt, p)
        return ex

    def tune_threshold(self, s1_local, p, y, n_true_eval, min_threshold):
        """Baseline exact macro-F0.5 sweep over decision candidates; tau selection rule of
        training/09_tune_threshold.py (tau >= min_threshold, highest tau among ties)."""
        thr, macro, n_pred, n_tp = self.metrics.exact_sweep(s1_local, p, y, n_true_eval)
        ok = thr >= min_threshold
        best = int(np.flatnonzero(ok)[np.argmax(macro[ok])])
        tau = float(thr[best])
        return tau, self.metrics.summarize(s1_local, p, y, n_true_eval, tau)

    def f05(self, tp, pred, n_true):
        return self.metrics.f05(tp, pred, n_true)

    # ------------------------------------------------------------------ output writer helpers
    def predict_script_helpers(self):
        """entity_ids() and grouped_lines() from training/11_predict_test.py (module name starts with a digit)."""
        path = self.src / "training" / "11_predict_test.py"
        spec = importlib.util.spec_from_file_location("baseline_predict_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.entity_ids, mod.grouped_lines


def load(cfg):
    return ClassicalBaseline(cfg)
