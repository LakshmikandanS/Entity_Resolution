"""Shared helpers for the neural experiment: config, streaming source reads, light text cleaning,
deterministic encoder split, resumable memmaps/progress files, and resource monitoring.

Nothing here re-implements baseline logic (normalisation, blocking, features, model, metric);
those come only through baseline_api.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

HERE = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):   # Windows consoles/pipes may default to cp1252; data has Indic text
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


# --------------------------------------------------------------------------------------- config
class Config(dict):
    """Dict with resolved paths: cfg.root, cfg.dataset, cfg.work."""

    @property
    def root(self) -> Path:
        return self["_root"]

    @property
    def dataset(self) -> Path:
        return self.root / self["paths"]["dataset"]

    @property
    def work(self) -> Path:
        return self.root / self["paths"]["work"]

    def wpath(self, *parts) -> Path:
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else HERE / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))
    root = (path.resolve().parent / cfg["paths"]["root"]).resolve()
    cfg["_root"] = root
    cfg["_config_path"] = str(path.resolve())
    # repo root on sys.path so the baseline packages (utils/, training/) import cleanly
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    cfg.work.mkdir(parents=True, exist_ok=True)
    return cfg


def add_config_arg(parser):
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    return parser


# --------------------------------------------------------------------------------------- sources
def source_path(cfg: Config, split: str, src: int) -> Path:
    return cfg.dataset / split / f"{split}_source{src}.tsv"


def ground_truth_path(cfg: Config) -> Path:
    return cfg.dataset / "train" / "train_ground_truth.tsv"


def count_rows(path: Path, cache_dir: Path) -> int:
    """Data rows in a TSV (header excluded). Streams bytes; cached by size+mtime."""
    st = path.stat()
    cache = cache_dir / "rowcounts.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if key in data:
        return data[key]
    n, last = 0, b"\n"
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            n += b.count(b"\n")
            last = b[-1:]
    if last != b"\n":
        n += 1
    data[key] = n - 1
    atomic_write_json(cache, data)
    return n - 1


def _csv_reader(path: Path, columns: list[str], block_bytes: int):
    ro = pacsv.ReadOptions(block_size=block_bytes, use_threads=False)
    po = pacsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False,
                            newlines_in_values=False)
    co = pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                              strings_can_be_null=False, quoted_strings_can_be_null=False)
    return pacsv.open_csv(path, read_options=ro, parse_options=po, convert_options=co)


def iter_source(path: Path, block_bytes: int = 1 << 24):
    """Yield (start_row, pyarrow.RecordBatch) over a source TSV. Row numbers follow file order and are
    the row ids used by every memmap in this experiment. Batch boundaries depend only on block_bytes."""
    reader = _csv_reader(path, SOURCE_COLUMNS, block_bytes)
    if reader.schema.names != SOURCE_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {reader.schema.names}, expected {SOURCE_COLUMNS}")
    start = 0
    for batch in reader:
        yield start, batch
        start += batch.num_rows


def iter_ground_truth(path: Path, block_bytes: int = 1 << 24):
    reader = _csv_reader(path, ["source1_entity_id", "matched_entity_ids"], block_bytes)
    for batch in reader:
        yield batch


def explode_ground_truth(batch: pa.RecordBatch) -> tuple[pa.Array, pa.Array]:
    """(s1_id, target_id) per link for one GT batch, vectorised."""
    import pyarrow.compute as pc

    lists = pc.split_pattern(batch.column("matched_entity_ids"), ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    s1 = pc.take(batch.column("source1_entity_id"), pc.filter(parent, keep))
    return s1, pc.filter(flat, keep)


# --------------------------------------------------------------------------------------- text
# Placeholders appear as whole comma-separated address components (", NULL,", ", n/a,"), verified on a
# 400k-row train sample; whole-field empties are plain empty strings. Matching is component-exact so
# words like "Thiruvananthapuram" are never touched.
_ADDR_PLACEHOLDERS = {"null", "<null>", "none", "n/a", "na", "nan", "nil", "-", "--", "n.a.", "not available"}
_WS = re.compile(r"\s+")
_INDIC = re.compile(r"[ऀ-ൿ]")  # Devanagari .. Malayalam (all scripts seen in train)


def clean_name(s: str) -> str:
    return _WS.sub(" ", s).strip()


def clean_address(s: str) -> str:
    parts = (_WS.sub(" ", p).strip() for p in s.split(","))
    return ", ".join(p for p in parts if p and p.lower() not in _ADDR_PLACEHOLDERS)


def has_indic(s: str) -> bool:
    return _INDIC.search(s) is not None


def stable_hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def hash_keys(keys) -> np.ndarray:
    return np.fromiter((stable_hash64(k) for k in keys), dtype=np.int64, count=len(keys))


# --------------------------------------------------------------------------------------- split
def _bucket(entity_id: str, salt: str) -> int:
    h = hashlib.blake2b((salt + "|" + entity_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % 10_000


def encoder_split_mask(entity_ids, cfg: Config) -> np.ndarray:
    """True for TRAIN S1 entities reserved for encoder fine-tuning (deterministic, id-hash based)."""
    salt, cut = cfg["split"]["salt"], int(round(cfg["split"]["encoder_frac"] * 10_000))
    return np.fromiter((_bucket(e, salt) < cut for e in entity_ids), dtype=bool, count=len(entity_ids))


def encoder_val_mask(entity_ids, cfg: Config) -> np.ndarray:
    salt, cut = cfg["split"]["salt"] + "|val", int(round(cfg["split"]["encoder_val_frac"] * 10_000))
    return np.fromiter((_bucket(e, salt) < cut for e in entity_ids), dtype=bool, count=len(entity_ids))


# --------------------------------------------------------------------------------------- files
def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def open_memmap(path: Path, shape: tuple, dtype, fill=None) -> np.memmap:
    """Create (or reopen r+) a .npy memmap of a fixed shape; refuses to reuse a file of another shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mm = np.lib.format.open_memmap(path, mode="r+")
        if mm.shape != tuple(shape) or mm.dtype != np.dtype(dtype):
            raise ValueError(f"{path}: existing {mm.shape}/{mm.dtype} != requested {shape}/{dtype}; "
                             f"delete it or the matching progress file to rebuild")
        return mm
    mm = np.lib.format.open_memmap(path, mode="w+", shape=tuple(shape), dtype=dtype)
    if fill is not None:
        mm[:] = fill
    return mm


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def gb(nbytes: float) -> float:
    return round(nbytes / 1e9, 3)


# --------------------------------------------------------------------------------------- resources
class Monitor:
    """Tracks per stage: wall time, peak RSS of this process, peak torch CUDA allocation of this process, and
    peak device-wide GPU memory in use (NVML; includes other processes and non-torch users such as
    XGBoost). RSS and device memory are sampled every 0.5 s. Writes a JSON log."""

    def __init__(self, cfg: Config, script: str):
        import psutil

        self._proc = psutil.Process()
        self.path = cfg.wpath("runlog", f"{script}.json")
        self.log = {"script": script, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
        self.peak_rss = 0
        self.peak_dev = 0
        self._nvml = self._nvml_handle()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._sample, daemon=True)
        self._t.start()
        self._t0 = time.time()

    @staticmethod
    def _nvml_handle():
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

    def _device_used(self) -> int:
        if self._nvml is None:
            return 0
        try:
            import pynvml

            return int(pynvml.nvmlDeviceGetMemoryInfo(self._nvml).used)
        except Exception:
            return 0

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            self.peak_dev = max(self.peak_dev, self._device_used())
            self._stop.wait(0.5)

    @staticmethod
    def _torch_cuda():
        """torch.cuda only if this process already uses CUDA (never creates a CUDA context just to measure)."""
        t = sys.modules.get("torch")
        try:
            return t.cuda if t is not None and t.cuda.is_available() and t.cuda.is_initialized() else None
        except Exception:
            return None

    @classmethod
    def cuda_peak(cls) -> int:
        c = cls._torch_cuda()
        return c.max_memory_allocated() if c is not None else 0

    @contextmanager
    def stage(self, name: str):
        t0, rss0, dev0 = time.time(), self.peak_rss, self.peak_dev
        self.peak_rss = self._proc.memory_info().rss
        self.peak_dev = self._device_used()
        c = self._torch_cuda()
        if c is not None:
            c.reset_peak_memory_stats()
        print(f"[{time.strftime('%H:%M:%S')}] >> {name}", flush=True)
        try:
            yield
        finally:
            rec = {"seconds": round(time.time() - t0, 1), "peak_rss_gb": gb(self.peak_rss),
                   "peak_vram_gb": gb(self.cuda_peak()), "peak_gpu_device_gb": gb(self.peak_dev)}
            self.log["stages"][name] = rec
            self.peak_rss = max(self.peak_rss, rss0)
            self.peak_dev = max(self.peak_dev, dev0)
            print(f"[{time.strftime('%H:%M:%S')}] << {name} {rec}", flush=True)
            self.flush()

    def note(self, key, value):
        self.log[key] = value
        self.flush()

    def flush(self):
        self.log["total_seconds"] = round(time.time() - self._t0, 1)
        atomic_write_json(self.path, self.log)

    def close(self):
        self._stop.set()
        self.flush()


def pick_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_vram(fraction: float):
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
