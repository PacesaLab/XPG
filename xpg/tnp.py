"""NumPy-style array functions on torch tensors (the subset the kernels use), so that code written against
``jax.numpy`` runs on PyTorch nearly unchanged: ``from . import tnp as jnp``.

Semantics follow jax.numpy with x64 enabled: float64 by default, Python scalars adopt the tensor's dtype,
``axis`` / ``keepdims`` keywords, functional (out-of-place) updates via ``index_add``.
"""
from __future__ import annotations

import numpy as np
import torch

import contextlib

from .backend import device as _backend_device

_CTX = {"dev": None}


def _device():
    return _CTX["dev"] or _backend_device()


def _f64():
    """Default float type: float64 (as jax.numpy with x64), float32 on devices without float64 (Apple MPS)."""
    return torch.float32 if _device().type == "mps" else torch.float64


@contextlib.contextmanager
def on(dev):
    """Run array creation on ``dev`` (e.g. the float32 image device for the integration kernels); on MPS every
    float64 request becomes float32."""
    old = _CTX["dev"]
    _CTX["dev"] = torch.device(dev) if dev is not None else None
    try:
        yield
    finally:
        _CTX["dev"] = old

float32, float64, int32, int64, bool_ = torch.float32, torch.float64, torch.int32, torch.int64, torch.bool
pi = float(np.pi)
fft = torch.fft
linalg = None   # replaced below

_NP2T = {np.dtype("float32"): torch.float32, np.dtype("float64"): torch.float64, np.dtype("int32"): torch.int32,
         np.dtype("int64"): torch.int64, np.dtype("int16"): torch.int16, np.dtype("bool"): torch.bool,
         np.dtype("uint8"): torch.uint8, np.dtype("int8"): torch.int8}


def _dt(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        t = dtype
    elif dtype is float:
        t = torch.float64
    elif dtype is int:
        t = torch.int64
    elif dtype is bool:
        t = torch.bool
    else:
        t = _NP2T[np.dtype(dtype)]
    return _f64() if t == torch.float64 else t


def _astype(self, dtype):
    return self.to(_dt(dtype))


if not hasattr(torch.Tensor, "astype"):     # numpy-style cast used throughout the kernels
    torch.Tensor.astype = _astype


def asarray(x, dtype=None):
    dt = _dt(dtype)
    if torch.is_tensor(x):
        if dt is None and x.dtype == torch.float64:
            dt = _f64()
        return x.to(_device(), dt) if dt is not None else x.to(_device())
    a = np.asarray(x)
    if dt is None:
        # as jax.numpy (x64): numpy arrays and numpy scalars keep their dtype (float32 image data stays float32);
        # Python floats and lists of them become float64
        if isinstance(x, (np.ndarray, np.generic)):
            dt = _NP2T.get(a.dtype, None)
        else:
            dt = torch.float64 if a.dtype.kind == "f" else _NP2T.get(a.dtype, None)
        if dt == torch.float64:
            dt = _f64()
    return torch.as_tensor(a, dtype=dt, device=_device())


array = asarray


def _t(x, like=None):
    """Tensor from x (scalars take the dtype / device of ``like``)."""
    if torch.is_tensor(x):
        return x
    if like is not None and torch.is_tensor(like):
        return torch.as_tensor(x, dtype=like.dtype if not isinstance(x, bool) else torch.bool, device=like.device)
    return asarray(x)


def zeros(shape, dtype=None):
    return torch.zeros(shape, dtype=_dt(dtype) or _f64(), device=_device())


def ones(shape, dtype=None):
    return torch.ones(shape, dtype=_dt(dtype) or _f64(), device=_device())


def zeros_like(x, dtype=None):
    return torch.zeros_like(x, dtype=_dt(dtype))


def ones_like(x, dtype=None):
    return torch.ones_like(x, dtype=_dt(dtype))


def arange(*args, dtype=None):
    return torch.arange(*args, dtype=_dt(dtype) or torch.int64, device=_device())


def eye(n, dtype=None):
    return torch.eye(n, dtype=_dt(dtype) or _f64(), device=_device())


def _red(fn, x, axis=None, keepdims=False):
    x = _t(x)
    if axis is None:
        out = fn(x)
        return out.reshape([1] * x.dim()) if keepdims else out
    return fn(x, dim=axis, keepdim=keepdims)


def sum(x, axis=None, keepdims=False, dtype=None):
    x = _t(x)
    if x.dtype == torch.bool:
        x = x.to(torch.int64)
    out = _red(torch.sum, x, axis, keepdims)
    return out.to(_dt(dtype)) if dtype is not None else out


def mean(x, axis=None, keepdims=False):
    x = _t(x)
    if not x.is_floating_point():
        x = x.to(_f64())
    return _red(torch.mean, x, axis, keepdims)


def max(x, axis=None, keepdims=False):
    x = _t(x)
    if axis is None:
        out = torch.max(x)
        return out.reshape([1] * x.dim()) if keepdims else out
    return torch.amax(x, dim=axis, keepdim=keepdims)


def min(x, axis=None, keepdims=False):
    x = _t(x)
    if axis is None:
        out = torch.min(x)
        return out.reshape([1] * x.dim()) if keepdims else out
    return torch.amin(x, dim=axis, keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    return _red(torch.any, _t(x).to(torch.bool), axis, keepdims)


def all(x, axis=None, keepdims=False):
    return _red(torch.all, _t(x).to(torch.bool), axis, keepdims)


def _scalar(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


# Python / numpy scalars go to torch as scalars: a device tensor per constant costs a blocking host -> device copy
# on every call (MPS: ~3 ms each, which dominated the integration kernels)

def maximum(a, b):
    if torch.is_tensor(a) and _scalar(b) and (a.is_floating_point() or isinstance(b, (int, np.integer))):
        return torch.clamp(a, min=float(b) if a.is_floating_point() else int(b))
    if torch.is_tensor(b) and _scalar(a) and (b.is_floating_point() or isinstance(a, (int, np.integer))):
        return torch.clamp(b, min=float(a) if b.is_floating_point() else int(a))
    if not torch.is_tensor(a):
        a, b = b, a
    return torch.maximum(a, _t(b, a))


def minimum(a, b):
    if torch.is_tensor(a) and _scalar(b) and (a.is_floating_point() or isinstance(b, (int, np.integer))):
        return torch.clamp(a, max=float(b) if a.is_floating_point() else int(b))
    if torch.is_tensor(b) and _scalar(a) and (b.is_floating_point() or isinstance(a, (int, np.integer))):
        return torch.clamp(b, max=float(a) if b.is_floating_point() else int(a))
    if not torch.is_tensor(a):
        a, b = b, a
    return torch.minimum(a, _t(b, a))


def where(c, a, b):
    c = _t(c)
    if torch.is_tensor(a) and _scalar(b):
        return torch.where(c, a, float(b) if a.is_floating_point() else int(b))
    if torch.is_tensor(b) and _scalar(a):
        return torch.where(c, float(a) if b.is_floating_point() else int(a), b)
    if not torch.is_tensor(a) and not torch.is_tensor(b):
        a, b = asarray(a), asarray(b)
    elif not torch.is_tensor(a):
        a = _t(a, b)
    elif not torch.is_tensor(b):
        b = _t(b, a)
    return torch.where(c, a, b)


def clip(x, lo=None, hi=None):
    x = _t(x)
    if lo is None and hi is None:
        return x
    if (lo is None or _scalar(lo)) and (hi is None or _scalar(hi)):
        cv = float if x.is_floating_point() else int          # keep integer tensors integer (index arithmetic)
        return torch.clamp(x, min=None if lo is None else cv(lo), max=None if hi is None else cv(hi))
    out = x
    if lo is not None:
        out = torch.maximum(out, _t(lo, x))
    if hi is not None:
        out = torch.minimum(out, _t(hi, x))
    return out


def broadcast_to(x, shape):
    return _t(x).expand(*shape) if _t(x).dim() else _t(x).expand(*shape)


def broadcast_arrays(*xs):
    return torch.broadcast_tensors(*[_t(x) for x in xs])


broadcast_shapes = torch.broadcast_shapes


def cross(a, b, axis=-1):
    a, b = torch.broadcast_tensors(_t(a), _t(b, a))
    return torch.linalg.cross(a, b, dim=axis)


def einsum(spec, *ops):
    return torch.einsum(spec, *[_t(o) for o in ops])


def dot(a, b):
    a, b = _t(a), _t(b)
    return a @ b


def take_along_axis(x, idx, axis):
    return torch.take_along_dim(_t(x), _t(idx), dim=axis)


def diff(x, n=1, axis=-1):
    return torch.diff(_t(x), n=n, dim=axis)


def round(x):
    return torch.round(_t(x))


def stack(xs, axis=0):
    return torch.stack([_t(x) for x in xs], dim=axis)


def concatenate(xs, axis=0):
    return torch.cat([_t(x) for x in xs], dim=axis)


def real(x):
    return torch.real(_t(x))


def index_add(arr, idx, values):
    """Functional ``arr.at[idx].add(values)`` (duplicate indices accumulate)."""
    idx = tuple(_t(i).to(torch.int64) for i in idx)
    shape = torch.broadcast_shapes(*[i.shape for i in idx], torch.as_tensor(values).shape)
    idx = tuple(i.expand(shape) for i in idx)
    v = _t(values, arr).to(arr.dtype).expand(shape)
    return arr.index_put(idx, v, accumulate=True)


def _unary(fn):
    return lambda x: fn(_t(x) if torch.is_tensor(x) or not np.isscalar(x) else torch.as_tensor(x, dtype=_f64(), device=_device()))


sqrt, exp, log, abs, floor, expm1 = (_unary(f) for f in (torch.sqrt, torch.exp, torch.log, torch.abs, torch.floor,
                                                        torch.expm1))
sin, cos, arccos, arctan2 = _unary(torch.sin), _unary(torch.cos), _unary(torch.arccos), torch.atan2
radians = _unary(torch.deg2rad)


class _Linalg:
    @staticmethod
    def norm(x, axis=None, keepdims=False):
        x = _t(x)
        if axis is None:
            return torch.linalg.vector_norm(x.reshape(-1))
        return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims)

    inv = staticmethod(torch.linalg.inv)
    solve = staticmethod(torch.linalg.solve)


linalg = _Linalg()


def erf(x):
    return torch.special.erf(_t(x))
