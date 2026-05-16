#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python -m utils.ensure_nltk_data

python -m compileall -q src
python -m pytest
