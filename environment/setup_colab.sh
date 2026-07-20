#!/usr/bin/env bash
set -euo pipefail

DRIVE_ROOT="/content/drive/MyDrive/var2026"
CUDA_EXT_CACHE="$DRIVE_ROOT/cuda_ext_cache"
DATASET_DRIVE_PATH="$DRIVE_ROOT/VAI_NVS_DATA_ROUND2"
REPO_DIR="$(pwd)"

# Drive must be mounted from an actual notebook cell BEFORE this script
# runs, not from here: google.colab.drive.mount() talks to the browser
# frontend through the notebook kernel's own process — calling it via
# `python3 -c "..."` from a `!bash ...` shell command spawns a separate
# subprocess that isn't the kernel, so it fails with
# "AttributeError: 'NoneType' object has no attribute 'kernel'".
if [ ! -d "/content/drive/MyDrive" ]; then
  echo "ERROR: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "  Run this in its own notebook cell FIRST, then re-run this script:" >&2
  echo "    from google.colab import drive" >&2
  echo "    drive.mount('/content/drive')" >&2
  exit 1
fi

mkdir -p "$CUDA_EXT_CACHE"

echo "== Linking dataset from Drive =="
# configs/scenes.yaml's dataset_root ("VAI_NVS_DATA_ROUND2") is resolved
# relative to the repo root at runtime, so the dataset must appear there —
# but the dataset itself lives on Drive (too large for git), so it's
# symlinked in rather than copied.
if [ ! -e "VAI_NVS_DATA_ROUND2" ]; then
  if [ -d "$DATASET_DRIVE_PATH" ]; then
    ln -s "$DATASET_DRIVE_PATH" VAI_NVS_DATA_ROUND2
  else
    echo "WARNING: dataset not found at $DATASET_DRIVE_PATH" >&2
    echo "  Upload the VAI_NVS_DATA_ROUND2/ folder to Drive at exactly that path before running the pipeline." >&2
  fi
fi

echo "== Installing Python dependencies =="
pip install -q -r environment/requirements.txt

echo "== Checking out submodule =="
git submodule update --init --recursive

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

restore_or_build () {
  local ext_name="$1"
  local ext_src_dir="third_party/gaussian-splatting/submodules/$ext_name"
  local py_name="${ext_name//-/_}"
  local cache_dir="$CUDA_EXT_CACHE/$ext_name"
  local cache_marker="$cache_dir.built"

  if [ -f "$cache_marker" ]; then
    echo "== Restoring cached $ext_name build =="
    cp -r "$cache_dir"/. "$SITE_PACKAGES/"
  else
    echo "== Building $ext_name from source (first run, slow) =="
    pip install -q "$ext_src_dir"
    # Copy the actual compiled artifacts (the .so plus install metadata)
    # out of site-packages into the Drive cache, instead of the old
    # `pip download --no-binary :all:` approach — these two submodules
    # don't package a round-trippable sdist, so that download silently
    # produced nothing to restore from on the next session.
    rm -rf "$cache_dir"
    mkdir -p "$cache_dir"
    cp -r "$SITE_PACKAGES/${py_name}"* "$cache_dir/"
    # Only mark the cache built after a successful copy — no more
    # `|| true` masking a failed/empty cache write.
    touch "$cache_marker"
  fi
}

restore_or_build diff-gaussian-rasterization
restore_or_build simple-knn

echo "== Setup complete =="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
