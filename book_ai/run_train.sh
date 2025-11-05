#!/usr/bin/env bash
set -euo pipefail

VENV_PY="/Users/kshitij.satish1/Library/CloudStorage/OneDrive-Sanas.ai/Desktop/Prj/book_ai/venv/bin/python"
SCRIPT="/Users/kshitij.satish1/Library/CloudStorage/OneDrive-Sanas.ai/Desktop/Prj/book_ai/src/train_cnn.py"

exec "$VENV_PY" "$SCRIPT" "$@"


