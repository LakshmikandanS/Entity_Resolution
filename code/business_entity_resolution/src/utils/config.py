"""Conservative defaults for an 8 GB laptop GPU and a ~5 GB practical RAM budget.

Every value can be overridden by an environment variable of the same name, and most scripts also
expose a CLI flag that wins over both.
"""
import os


def _env(name, default, cast):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return cast(value)


# Memory / device budgets
RAM_BUDGET_GB = _env("RAM_BUDGET_GB", 5.0, float)          # practical, not the theoretical 16 GB
MAX_GPU_MEMORY_GB = _env("MAX_GPU_MEMORY_GB", 6.0, float)  # leaves ~2 GB for CUDA context/driver
N_THREADS = _env("N_THREADS", 4, int)

# Streaming / batching
CHUNK_ROWS = _env("CHUNK_ROWS", 200_000, int)              # records per streamed chunk (01, 03)
BATCH_SIZE = _env("BATCH_SIZE", 65_536, int)               # candidate pairs per GPU feature batch (05)
PREDICT_BATCH_ROWS = _env("PREDICT_BATCH_ROWS", 1_000_000, int)  # rows per model.predict batch
TRAIN_ITER_ROWS = _env("TRAIN_ITER_ROWS", 500_000, int)    # rows per XGBoost DataIter batch

# Blocking
MAX_CANDIDATES_PER_S1 = _env("MAX_CANDIDATES_PER_S1", 32, int)
MAX_KEY_FREQUENCY = _env("MAX_KEY_FREQUENCY", 1000, int)            # rare families P, C, X, S
MAX_KEY_FREQUENCY_COMMON = _env("MAX_KEY_FREQUENCY_COMMON", 1000, int)  # frequent families N, A
MAX_EXPANDED_POSTINGS = _env("MAX_EXPANDED_POSTINGS", 8_000_000, int)
S1_PER_SHARD = _env("S1_PER_SHARD", 50_000, int)
INDEX_PARTITIONS = _env("INDEX_PARTITIONS", 16, int)       # power of two

# Modelling
N_FOLDS = _env("N_FOLDS", 2, int)
SEED = _env("SEED", 2026, int)


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
