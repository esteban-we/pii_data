#!/bin/sh
# One command per checkout: point git at the tracked hooks and fetch the bytes.
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath = $(git config --get core.hooksPath)"
python3 data/oss_sync.py pull "$@"
