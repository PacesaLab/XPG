#!/usr/bin/env bash
# XPG installer for Linux and macOS, on a laptop or an HPC cluster:
#
#   git clone https://github.com/PacesaLab/XPG.git && cd XPG && bash install.sh [options]
#
# It creates a Python environment in the checkout with the newest PyTorch build for this machine, installs XPG, links
# the `xpg` command into ~/.local/bin and runs a smoke test. To update: git pull, then run it again. On a cluster
# (checkout on scratch, GPU smoke test via SLURM): bash install.sh --srun "--gpus=1 --time=00:10:00"
#
# Without a checkout, `curl -fsSL https://raw.githubusercontent.com/PacesaLab/XPG/main/install.sh | bash` fetches XPG
# into ~/xpg first (git, or a tarball without git).
#
#   --dir DIR                where to fetch XPG when run through curl (default: ~/xpg)
#   --prefix DIR             environment location (default: <source>/.venv)
#   --cpu                    force the CPU build (default: CUDA if an NVIDIA driver is present; macOS: Apple GPU)
#   --cuda 126|128|130       force a PyTorch CUDA build
#   --rocm 6.4               AMD GPUs: PyTorch ROCm build
#   --xpu                    Intel GPUs: PyTorch XPU build
#   --python EXE             interpreter to build the environment from (Python >= 3.10)
#   --srun "SLURM OPTIONS"   run the smoke test on a compute node (login nodes have no GPU)
#   --dev                    also install pytest and run the test suite
#   --no-test                skip the smoke test
#   --no-link                do not link `xpg` into ~/.local/bin
#   --openfold3              also install OpenFold3 (Apache-2.0) into <source>/.venv-openfold3 and download its weights
#                            (~2 GB, into ~/.openfold3): `xpg mr` then predicts the search model when none is given
#
# Only an NVIDIA driver is needed for CUDA (the wheels ship the runtime: cu130 needs driver >= 580, cu128 >= 570,
# cu126 >= 525).
set -euo pipefail

REPO="${XPG_REPO:-https://github.com/PacesaLab/XPG}"     # XPG_REPO: a fork or mirror
DIR=""
PREFIX=""
PYTHON=""
FORCE_CPU=0
CUDA=""
ROCM=""
XPU=0
DEV=0
TEST=1
LINK=1
OF3=0
SRUN=""
here=""
src="${BASH_SOURCE[0]:-}"
if [ -n "$src" ] && [ -f "$(dirname "$src")/pyproject.toml" ]; then here="$(cd "$(dirname "$src")" && pwd)"; fi

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --cpu) FORCE_CPU=1; shift;;
    --cuda) CUDA="$2"; shift 2;;
    --rocm) ROCM="$2"; shift 2;;
    --xpu) XPU=1; shift;;
    --dev) DEV=1; shift;;
    --no-test) TEST=0; shift;;
    --no-link) LINK=0; shift;;
    --openfold3) OF3=1; shift;;
    --srun) SRUN="$2"; shift 2;;
    -h|--help) sed -n '2,/^set -euo/p' "${src:-/dev/null}" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

say() { printf '\033[1m[xpg-install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[xpg-install] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- source: this checkout, or fetched (curl | bash) ----------------------------------------------------------------
if [ -z "$here" ]; then
  here="${DIR:-$HOME/xpg}"
  if [ -d "$here/.git" ]; then
    say "updating $here"; git -C "$here" pull --ff-only
  elif command -v git >/dev/null 2>&1; then
    say "fetching XPG into $here"; git clone --depth 1 "$REPO" "$here"
  else
    say "fetching XPG into $here (no git: tarball)"; mkdir -p "$here"
    curl -fsSL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$here" --strip-components 1
  fi
fi
PREFIX="${PREFIX:-$here/.venv}"

# ---- Python interpreter (>= 3.10) ------------------------------------------------------------------------------
pick_python() {
  local c
  for c in "$PYTHON" python3.13 python3.12 python3.11 python3.10 python3; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# ---- accelerator --------------------------------------------------------------------------------------------------
OS="$(uname -s)"; ARCH="$(uname -m)"
TORCH_INDEX=""     # PyTorch wheel index: "" = PyPI (macOS), .../whl/cpu, .../whl/cuXXX
cuda_for_driver() {   # newest PyTorch CUDA build supported by an NVIDIA driver version
  local major="${1%%.*}"
  if [ "${major:-0}" -ge 580 ]; then echo 130; elif [ "${major:-0}" -ge 570 ]; then echo 128
  elif [ "${major:-0}" -ge 525 ]; then echo 126; fi
}
CU=""
if [ "$FORCE_CPU" = 1 ]; then
  CU=""
elif [ -n "$ROCM" ] || [ "$XPU" = 1 ]; then
  CU=""
elif [ -n "$CUDA" ]; then
  CU="$CUDA"
elif [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  CU="$(cuda_for_driver "$drv")"
  if [ -n "$CU" ]; then say "NVIDIA driver $drv -> PyTorch cu$CU"; else
    say "NVIDIA driver $drv is too old for current PyTorch CUDA builds (needs >= 525): installing the CPU build"; fi
elif [ "$OS" = "Linux" ] && command -v sinfo >/dev/null 2>&1; then
  # SLURM login node without a visible GPU: the compute nodes may have one. Ask the driver there if possible.
  if [ -n "$SRUN" ]; then
    drv="$(srun $SRUN nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    CU="$(cuda_for_driver "$drv")"
    [ -n "$CU" ] && say "compute-node NVIDIA driver $drv -> PyTorch cu$CU"
  fi
  [ -z "$CU" ] && say "SLURM login node, no GPU visible: installing the CPU build (use --cuda 130 or --srun to target GPU nodes)"
else
  if [ "$OS" = "Linux" ] && { command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; }; then
    say "AMD ROCm tools found: rerun with --rocm <version> (e.g. --rocm 6.4) for the GPU build; installing the CPU build"
  elif [ "$OS" = "Linux" ] && { command -v xpu-smi >/dev/null 2>&1 || command -v sycl-ls >/dev/null 2>&1; }; then
    say "Intel GPU tools found: rerun with --xpu for the GPU build; installing the CPU build"
  else
    say "no NVIDIA GPU detected ($OS $ARCH): installing the CPU build"
  fi
fi
if [ -n "$ROCM" ]; then TORCH_INDEX="https://download.pytorch.org/whl/rocm$ROCM"; say "PyTorch ROCm $ROCM build"
elif [ "$XPU" = 1 ]; then TORCH_INDEX="https://download.pytorch.org/whl/xpu"; say "PyTorch XPU build"
elif [ -n "$CU" ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu$CU"
elif [ "$OS" = "Linux" ]; then TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi

# ---- environment -------------------------------------------------------------------------------------------------
PY="$(pick_python)" || die "Python >= 3.10 not found (use --python, or install e.g. miniforge / uv)"
say "Python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
PKGS=(numpy scipy gemmi h5py hdf5plugin lz4 matplotlib)
TORCH_ARGS=(torch); [ -n "$TORCH_INDEX" ] && TORCH_ARGS=(torch --index-url "$TORCH_INDEX")
[ "$DEV" = 1 ] && PKGS+=(pytest)

if command -v uv >/dev/null 2>&1; then
  say "creating $PREFIX with uv"
  uv venv --python "$PY" "$PREFIX" >/dev/null
  # shellcheck disable=SC1091
  . "$PREFIX/bin/activate"
  uv pip install --upgrade "${TORCH_ARGS[@]}"
  uv pip install --upgrade "${PKGS[@]}"
  uv pip install -e "$here" --no-deps
else
  say "creating $PREFIX with venv / pip"
  if ! "$PY" -m venv "$PREFIX" >/dev/null 2>&1; then
    rm -rf "$PREFIX"
    # system Pythons without ensurepip (Debian/Ubuntu without python3-venv, many clusters): use a conda-forge
    # Python from an environment module or an existing conda installation instead
    alt=""
    if type module >/dev/null 2>&1; then
      for m in miniforge3 mamba miniconda3 anaconda3 python; do
        if module load "$m" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1 \
           && python3 -c 'import ensurepip' >/dev/null 2>&1; then alt="$(command -v python3)"; say "using module $m"; break; fi
      done
    fi
    if [ -z "$alt" ]; then
      for c in mamba conda; do
        if command -v "$c" >/dev/null 2>&1; then
          base="$("$c" info --base 2>/dev/null || true)"
          [ -x "$base/bin/python3" ] && alt="$base/bin/python3" && break
        fi
      done
    fi
    [ -n "$alt" ] || die "'$PY -m venv' failed (no ensurepip) and no conda / miniforge Python found: install uv
    (https://docs.astral.sh/uv/) or python3-venv, or pass --python <interpreter with venv support>"
    say "Python: $("$alt" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
    "$alt" -m venv "$PREFIX"
  fi
  # shellcheck disable=SC1091
  . "$PREFIX/bin/activate"
  python -m pip install -q --upgrade pip
  python -m pip install --upgrade "${TORCH_ARGS[@]}"
  python -m pip install --upgrade "${PKGS[@]}"
  python -m pip install -e "$here" --no-deps
fi
python - <<'EOF'
import importlib.metadata as md
vs = []
for p in ("torch", "numpy", "scipy", "gemmi", "h5py", "hdf5plugin"):
    try:
        vs.append(f"{p} {md.version(p)}")
    except md.PackageNotFoundError:
        pass
print("[xpg-install] installed:", ", ".join(vs))
EOF

# ---- smoke test --------------------------------------------------------------------------------------------------
if [ "$TEST" = 1 ]; then
  SMOKE='import torch, xpg
from xpg.backend import device, describe
d = device()
x = torch.ones((2048, 2048), dtype=torch.float64, device=d)
v = float((x @ x).sum())
assert v == 2048.0 ** 3, v
print("[xpg-install] smoke test OK:", describe())'
  if [ -n "$SRUN" ]; then
    say "smoke test on a compute node: srun $SRUN"
    srun $SRUN bash -c ". '$PREFIX/bin/activate' && python -c '$SMOKE'"
  else
    python -c "$SMOKE" || die "smoke test failed (on a cluster login node use --srun \"--gpus=1 --time=00:10:00\")"
  fi
  if [ "$DEV" = 1 ]; then
    say "running the test suite"
    if [ -n "$SRUN" ]; then srun $SRUN bash -c ". '$PREFIX/bin/activate' && cd '$here' && python -m pytest -q tests"
    else (cd "$here" && python -m pytest -q tests); fi
  fi
fi

# ---- optional: OpenFold3 for predicted MR search models (own environment: its pins differ from XPG's) -------------
if [ "$OF3" = 1 ]; then
  OF3_PREFIX="$here/.venv-openfold3"
  say "installing OpenFold3 into $OF3_PREFIX"
  # biotite 1.4: newer releases make AtomArray a Sequence, which breaks OpenFold3 0.5's batch transfer
  OF3_PKGS=(openfold3 "biotite==1.4.0")
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PY" "$OF3_PREFIX" >/dev/null
    VIRTUAL_ENV="$OF3_PREFIX" uv pip install --python "$OF3_PREFIX/bin/python" "${TORCH_ARGS[@]}"
    VIRTUAL_ENV="$OF3_PREFIX" uv pip install --python "$OF3_PREFIX/bin/python" "${OF3_PKGS[@]}"
  else
    python -m venv "$OF3_PREFIX"
    "$OF3_PREFIX/bin/python" -m pip install -q --upgrade pip
    "$OF3_PREFIX/bin/python" -m pip install "${TORCH_ARGS[@]}"
    "$OF3_PREFIX/bin/python" -m pip install "${OF3_PKGS[@]}"
  fi
  say "downloading the OpenFold3 weights (~2 GB) into ~/.openfold3"
  "$OF3_PREFIX/bin/setup_openfold" --non-interactive
fi

if [ "$LINK" = 1 ]; then
  mkdir -p "$HOME/.local/bin" && ln -sf "$PREFIX/bin/xpg" "$HOME/.local/bin/xpg"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) say "xpg linked into ~/.local/bin";;
    *) say "xpg linked into ~/.local/bin; add it to your PATH:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc";;
  esac
fi
say "done. Run:  xpg --images /path/to/master.h5 -o run1"
