"""RAM / VRAM reporting, resource guards and device selection."""
import gc
import importlib
import os
import platform
import subprocess

import psutil

GB = 2**30


def ram_report():
    vm = psutil.virtual_memory()
    rss = psutil.Process(os.getpid()).memory_info().rss
    return {"total_gb": vm.total / GB, "available_gb": vm.available / GB, "process_rss_gb": rss / GB}


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def gpu_report():
    torch = _torch()
    info = {"torch": getattr(torch, "__version__", None), "cuda_available": False}
    if torch is not None and torch.cuda.is_available():
        dev = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(dev)
        props = torch.cuda.get_device_properties(dev)
        info.update({
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_gb": total / GB,
            "free_gb": free / GB,
            "allocated_gb": torch.cuda.memory_allocated(dev) / GB,
            "reserved_gb": torch.cuda.memory_reserved(dev) / GB,
            "max_allocated_gb": torch.cuda.max_memory_allocated(dev) / GB,
        })
    else:
        try:  # driver-level view even when torch has no CUDA
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()
            if out:
                name, total, used = [x.strip() for x in out.splitlines()[0].split(",")]
                info.update({"name": name, "total_gb": float(total) / 1024, "used_gb": float(used) / 1024})
        except Exception:
            pass
    return info


def short_mem_string():
    r = ram_report()
    s = f"RSS {r['process_rss_gb']:.2f}G avail {r['available_gb']:.1f}G"
    torch = _torch()
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        s += (f" | VRAM alloc {torch.cuda.memory_allocated() / GB:.2f}G"
              f" peak {torch.cuda.max_memory_allocated() / GB:.2f}G")
    return s


def check_ram(required_gb, what, force=False):
    """Fail before an expensive stage if the estimated peak does not fit in available RAM."""
    from .io import fail, log
    avail = ram_report()["available_gb"]
    log(f"resource check [{what}]: estimated peak {required_gb:.2f} GB, available {avail:.2f} GB")
    if required_gb > avail * 0.9:
        msg = (f"[{what}] needs ~{required_gb:.2f} GB RAM but only {avail:.2f} GB is available. "
               f"Close other programs, lower the chunk/batch parameters, or pass --force.")
        if force:
            log("WARNING: " + msg + " (continuing because of --force)")
        else:
            fail(msg)


def get_device(prefer="cuda", max_gpu_gb=None):
    """Return a torch.device and cap this process's share of VRAM."""
    torch = _torch()
    if prefer == "cuda" and torch is not None and torch.cuda.is_available():
        dev = torch.device("cuda")
        if max_gpu_gb:
            total = torch.cuda.get_device_properties(0).total_memory / GB
            frac = max(0.1, min(0.95, max_gpu_gb / total))
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        torch.cuda.reset_peak_memory_stats()
        return dev
    if prefer == "cuda":
        from .io import log
        log("WARNING: CUDA not available, falling back to CPU tensors (much slower)")
    return torch.device("cpu") if torch is not None else None


def release_gpu():
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def xgboost_status():
    """(importable, version, built_with_cuda)"""
    try:
        xgb = importlib.import_module("xgboost")
    except Exception:
        return False, None, False
    try:
        cuda = bool(xgb.build_info().get("USE_CUDA", False))
    except Exception:
        cuda = False
    return True, xgb.__version__, cuda


def library_versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pyarrow", "sklearn", "torch", "xgboost", "psutil", "joblib"]:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = None
    return out
