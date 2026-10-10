#!/bin/sh
# Set up the scoring server inside this repo copy: Python, a venv, the three packages
# and the model the benchmark picked. Needs uv (https://docs.astral.sh/uv/). Nothing
# is written outside the repo folder. Safe to run again: what is there is kept.
set -e
cd "$(dirname "$0")/.."
export UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_PREFERENCE=only-managed
REVISION=af3c52e6a9d8bccc8e8c3ed4ba9d7fa084a47e33
# macOS and Linux put the venv's Python in bin/, Windows (run this from Git Bash) in Scripts/.
venv_python() { for p in .venv/bin/python .venv/Scripts/python.exe; do [ -x "$p" ] && echo "$p" && return; done; }
[ -n "$(venv_python)" ] || uv venv --python 3.12 .venv
PY=$(venv_python)
uv pip install --python .venv onnxruntime tokenizers numpy
if [ -f models/horizon-guard-base/model_quantized.onnx ] && grep -q "$REVISION" models/horizon-guard-base/REVISION 2>/dev/null; then
    echo "Horizon-Labs model already at $REVISION"
else
    "$PY" scripts/fetch_model.py Horizon-Labs/prompt-injection-guard-base models/horizon-guard-base \
        onnx/model_quantized.onnx tokenizer.json config.json --revision "$REVISION"
fi
echo
echo "Start it with:"
echo "  $PY scripts/server.py --model models/horizon-guard-base --host <address clients reach> --port 8765"
