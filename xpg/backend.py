"""PyTorch backend: device / dtype selection and numpy <-> torch helpers.

Devices: CUDA when available (NVIDIA; AMD ROCm builds of PyTorch appear as CUDA too), else an Intel GPU (XPU) with
float64 support, else the CPU (XPG_DEVICE=cpu|cuda|cuda:N|xpu|mps overrides). Apple MPS (and Intel GPUs without
float64) run only the float32 image kernels (``image_device``: spot finding, forward-model kernels where they pay);
float64 work (refinement, scaling, indexing, integration) always runs on a device with float64 support (``device``).
"""
from __future__ import annotations

import contextlib
import os
import threading

import numpy as np
import torch

F64 = torch.float64
F32 = torch.float32
I32 = torch.int32
I64 = torch.int64


def _xpu_available() -> bool:
    try:
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:  # pragma: no cover (older / unusual builds)
        return False


def _xpu_has_fp64() -> bool:
    try:
        return bool(getattr(torch.xpu.get_device_properties(0), "has_fp64", False))
    except Exception:  # pragma: no cover
        return False


def _pick_device() -> torch.device:
    want = os.environ.get("XPG_DEVICE", "").strip().lower()
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _xpu_available() and _xpu_has_fp64():
        return torch.device("xpu")
    return torch.device("cpu")


_DEVICE = _pick_device()
_IMAGE_DEVICE = _DEVICE
if _DEVICE.type == "cpu" and not os.environ.get("XPG_DEVICE"):
    if _xpu_available():
        _IMAGE_DEVICE = torch.device("xpu")  # Intel GPU without float64: float32 image kernels only
    elif torch.backends.mps.is_available():
        _IMAGE_DEVICE = torch.device("mps")  # Apple GPU for the float32 image kernels (XPG_DEVICE=cpu: CPU only)
if _DEVICE.type == "mps":            # float64 is not supported on MPS: double-precision work stays on the CPU
    _DEVICE = torch.device("cpu")


def available_cpus() -> int:
    """CPUs this process may use (SLURM / cgroup affinity), not the node's total."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 2


if _DEVICE.type == "cpu" and "XPG_TORCH_THREADS" not in os.environ:
    torch.set_num_threads(max(1, available_cpus()))
elif "XPG_TORCH_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["XPG_TORCH_THREADS"]))


_TL = threading.local()


def device() -> torch.device:
    """Device for float64 work (a thread-local override from float64_on() first)."""
    return getattr(_TL, "dev", None) or _DEVICE


def torch_threads(n):
    """Temporarily n intra-op CPU threads (None: no change): tiny tensors (per-block refinement, a few hundred
    spots) are dominated by thread-pool synchronisation with many threads (cluster: predict_xyphi_t 1.7 ms per call
    with 16 threads)."""
    prev = torch.get_num_threads()
    if n is not None:
        torch.set_num_threads(int(n))
    try:
        yield
    finally:
        if n is not None:
            torch.set_num_threads(prev)


torch_threads = contextlib.contextmanager(torch_threads)


@contextlib.contextmanager
def float64_on(dev):
    """Run the float64 work of this thread on dev (None: no change). Small least-squares problems (per-block
    refinement: a few hundred spots) are launch- and sync-bound on a GPU and run faster on the CPU."""
    prev = getattr(_TL, "dev", None)
    _TL.dev = torch.device(dev) if dev is not None else prev
    try:
        yield
    finally:
        _TL.dev = prev


def image_device() -> torch.device:
    """Device for float32 image kernels (may be MPS)."""
    return _IMAGE_DEVICE


def describe() -> str:
    d = _IMAGE_DEVICE
    if d.type == "cuda":
        rt = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else "CUDA"
        return f"torch {torch.__version__}, {rt} {torch.cuda.get_device_name(d)}"
    if d.type == "xpu":
        try:
            name = torch.xpu.get_device_name(d)
        except Exception:  # pragma: no cover
            name = "Intel GPU"
        return (f"torch {torch.__version__}, XPU {name}" + ("" if _DEVICE.type == "xpu" else
                f" for image kernels, CPU ({torch.get_num_threads()} threads) for float64 work"))
    if d.type == "mps":
        return (f"torch {torch.__version__}, Apple GPU (MPS) for image kernels, CPU ({torch.get_num_threads()} "
                f"threads) for float64 work")
    return f"torch {torch.__version__}, {d.type} ({torch.get_num_threads()} threads)"


def t64(x, dev=None) -> torch.Tensor:
    """float64 tensor on the compute device (no copy if already there)."""
    return torch.as_tensor(np.asarray(x, np.float64) if not torch.is_tensor(x) else x, dtype=F64,
                           device=dev or device())


def np_(x) -> np.ndarray:
    """numpy array from a tensor (or pass-through)."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def segment_sum(values: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    """Sum of ``values`` per segment id (0..n-1) along the first axis (jax.ops.segment_sum)."""
    out = torch.zeros((n,) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
    return out.index_add_(0, ids.to(values.device, I64), values)


def compiled(fn, name="kernel", dev=None, devices=("cpu", "mps"), dynamic=False):
    """``torch.compile`` (fuses the elementwise passes of an image kernel: 2x CPU, 6x MPS) with a fallback to eager
    execution if compilation is unavailable or fails (no C++ compiler / Triton). Only for ``devices``: on CUDA the
    eager kernels are already memory-bound at a few ms per frame and compiling (10-200 s on a busy node) does not pay.
    XPG_COMPILE=0 disables, =1 forces. Callers keep input shapes fixed (pad batches): one compilation per run;
    dynamic=True for kernels called with varying batch sizes (one compilation per remaining static shape)."""
    mode = os.environ.get("XPG_COMPILE", "").strip().lower()
    if mode in ("0", "false", "no"):
        return fn
    if mode not in ("1", "true", "yes") and (dev or _IMAGE_DEVICE).type not in devices:
        return fn
    state = {"f": None}

    def run(*a, **k):
        if state["f"] is None:
            try:
                f = torch.compile(fn, dynamic=dynamic)
                out = f(*a, **k)
                state["f"] = f
                return out
            except Exception as e:          # noqa: BLE001 - any compiler failure: run eagerly
                import warnings
                warnings.warn(f"torch.compile failed for {name} ({type(e).__name__}: {str(e)[:200]}); running eagerly")
                state["f"] = fn
        return state["f"](*a, **k)

    return run
