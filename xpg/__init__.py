"""XPG (X-tal data Processing on GPU): rotation-data processing for macromolecular crystallography (PyTorch).

Compute device: CUDA / ROCm / Intel XPU / Apple GPU when available, else the CPU (XPG_DEVICE overrides; see
backend.py). Image-decoding worker processes (XPG_DECODE_WORKER=1) import only the I/O modules.
"""
__version__ = "0.2.0"
