"""Paths, streaming readers, atomic writers, manifests and logging."""
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

SOURCES = ("s1", "s2", "s3")
TARGET_SOURCES = ("s2", "s3")
_T0 = time.time()


# --------------------------------------------------------------------------- logging
def log(msg, mem=True):
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}"
    if mem:
        from .gpu import short_mem_string
        line += f"  | {short_mem_string()}"
    print(line, flush=True)


def fail(msg, code=2):
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


# --------------------------------------------------------------------------- paths
class Paths:
    """All on-disk locations. `work` holds bulky intermediates, `artifacts` holds small,
    reviewable outputs (rewrite map, threshold, manifests, model bundle)."""

    def __init__(self, data_dir, work_dir, artifacts_dir, split="train"):
        self.data_dir = os.path.abspath(data_dir)
        self.work = os.path.abspath(work_dir)
        self.artifacts = os.path.abspath(artifacts_dir)
        self.split = split

    def raw(self, source):
        return os.path.join(self.data_dir, self.split, f"{self.split}_source{source[-1]}.tsv")

    @property
    def ground_truth(self):
        return os.path.join(self.data_dir, "train", "train_ground_truth.tsv")

    def d(self, *parts, split=None):
        path = os.path.join(self.work, split or self.split, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    normalized = property(lambda self: self.d("normalized"))
    canon = property(lambda self: self.d("canon"))
    records = property(lambda self: self.d("records"))
    index = property(lambda self: self.d("index"))
    candidates = property(lambda self: self.d("candidates"))
    features_base = property(lambda self: self.d("features_base"))
    features = property(lambda self: self.d("features"))
    predictions = property(lambda self: self.d("predictions"))

    @property
    def labels(self):
        return self.d("labels", split="train")

    @property
    def trainset(self):
        return self.d("trainset", split="train")

    @property
    def oof(self):
        return self.d("oof", split="train")

    @property
    def models(self):
        path = os.path.join(self.work, "models")
        os.makedirs(path, exist_ok=True)
        return path

    def artifact(self, name):
        os.makedirs(self.artifacts, exist_ok=True)
        return os.path.join(self.artifacts, name)


def add_common_args(parser, split=True):
    from . import config
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    parser.add_argument("--data-dir", default=os.path.join(root, "dataset"),
                        help="folder containing train/ and test/")
    parser.add_argument("--work-dir", default=os.path.join(root, "work"),
                        help="bulky intermediates (several GB)")
    parser.add_argument("--artifacts-dir", default=os.path.join(root, "artifacts"),
                        help="small reviewable outputs")
    if split:
        parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n-threads", type=int, default=config.N_THREADS)
    parser.add_argument("--force", action="store_true",
                        help="ignore resource-check failures and rebuild finished stages")
    return parser


def paths_from_args(args):
    return Paths(args.data_dir, args.work_dir, args.artifacts_dir, getattr(args, "split", "train"))


def limit_threads(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    pa.set_cpu_count(n)
    pa.set_io_thread_count(max(1, min(n, 4)))


# --------------------------------------------------------------------------- manifests
def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


MANIFEST = "_MANIFEST.json"


def stage_done(directory):
    return os.path.exists(os.path.join(directory, MANIFEST))


def write_manifest(directory, stats):
    stats = dict(stats)
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(os.path.join(directory, MANIFEST), stats)


def read_manifest(directory, what=""):
    path = os.path.join(directory, MANIFEST)
    if not os.path.exists(path):
        fail(f"{what or directory} is not finished (no {MANIFEST} in {directory}); run the earlier stage first")
    return read_json(path)


def clear_stage(directory):
    for p in glob.glob(os.path.join(directory, "*")):
        if os.path.isfile(p):
            os.remove(p)


SIGNATURE = "_INPUTS.json"


def upstream_stamp(directory):
    """Finish time recorded in an upstream stage's manifest (changes whenever it is rebuilt)."""
    p = os.path.join(directory, MANIFEST)
    return read_json(p).get("finished_at") if os.path.exists(p) else None


def file_stamp(path):
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def begin_stage(directory, signature, force=False):
    """Decide whether a stage can be skipped or resumed, given what its outputs were built from.

    Returns True when the stage is finished AND was built from the same inputs/settings (skip it).
    Otherwise partial or stale outputs are deleted when the signature differs (or with --force),
    so a resumed run never mixes shards made under different settings. A finished stage written
    before signatures existed is adopted as-is."""
    sig_path = os.path.join(directory, SIGNATURE)
    sig = json.loads(json.dumps(signature, default=_json_default, sort_keys=True))
    old = read_json(sig_path) if os.path.exists(sig_path) else None
    has_output = any(os.path.isfile(p) and os.path.basename(p) != SIGNATURE
                     for p in glob.glob(os.path.join(directory, "*")))
    done = stage_done(directory)
    if not force and done and old in (None, sig):
        if old is None:
            write_json(sig_path, sig)
        log(f"{directory} already finished with the same inputs (use --force to rebuild)")
        return True
    if force or (has_output and old != sig):
        if has_output:
            log(f"{directory}: {'--force' if force else 'inputs or settings changed'}; clearing previous outputs")
        clear_stage(directory)
    write_json(sig_path, sig)
    return False


# --------------------------------------------------------------------------- raw TSV
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def iter_raw_tsv(path, chunk_rows):
    """Stream a challenge TSV as Arrow record batches of string columns.

    No quoting (names contain quotes and commas), no null conversion: placeholders such as
    'None' or '<NULL>' stay literal strings and are handled by the normaliser."""
    block = max(1 << 20, int(chunk_rows * 110))  # ~110 bytes/row -> ~chunk_rows per block
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block, use_threads=True),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, double_quote=False,
                                         newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in RAW_COLUMNS
                                                           + ["source1_entity_id", "matched_entity_ids"]},
                                             null_values=[], strings_can_be_null=False,
                                             quoted_strings_can_be_null=False),
    )
    for batch in reader:
        yield batch


def numeric_ids(id_array, prefix):
    """'S2-000123' -> 123 (int64). Validates the prefix so a wrong file fails loudly."""
    ids = id_array.to_pylist() if hasattr(id_array, "to_pylist") else list(id_array)
    out = np.empty(len(ids), dtype=np.int64)
    for i, s in enumerate(ids):
        if not s.startswith(prefix):
            fail(f"unexpected id {s!r}, expected prefix {prefix!r}")
        out[i] = int(s[len(prefix):])
    return out


# --------------------------------------------------------------------------- parquet
def shard_name(i):
    return f"part-{i:05d}.parquet"


def list_shards(directory):
    return sorted(glob.glob(os.path.join(directory, "part-*.parquet")))


class AtomicParquetWriter:
    """Writes row groups incrementally to <path>.tmp and renames on close()."""

    def __init__(self, path, schema=None, compression="zstd"):
        self.path = path
        self.tmp = path + ".tmp"
        self.schema = schema
        self.compression = compression
        self.writer = None
        self.rows = 0

    def write_table(self, table):
        if self.writer is None:
            self.schema = self.schema or table.schema
            self.writer = pq.ParquetWriter(self.tmp, self.schema, compression=self.compression)
        self.writer.write_table(table.cast(self.schema) if table.schema != self.schema else table)
        self.rows += table.num_rows

    def write_columns(self, columns: dict):
        self.write_table(pa.table({k: pa.array(v) for k, v in columns.items()}))

    def close(self):
        if self.writer is None:  # empty shard: still produce a valid file
            if self.schema is None:
                raise RuntimeError(f"no schema for empty shard {self.path}")
            pq.write_table(self.schema.empty_table(), self.tmp, compression=self.compression)
        else:
            self.writer.close()
        os.replace(self.tmp, self.path)


def iter_parquet(paths, columns=None, batch_rows=1_000_000):
    """Yield (path, pyarrow.RecordBatch) for each row-group-sized batch."""
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield path, batch


def parquet_rows(paths):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def batch_numpy(batch, name, dtype=None):
    arr = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def disk_size_gb(paths):
    return sum(os.path.getsize(p) for p in paths) / 2**30
