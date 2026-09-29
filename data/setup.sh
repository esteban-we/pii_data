#!/bin/sh
# One command per checkout of pii_data: bring the metadata up to date, point git at the
# tracked hooks, and fetch the bytes the index names. Idempotent, so rerun it any time.
#
# After it, two commands are enough: `git pull` brings new metadata and, through
# .githooks/post-merge, the bytes it names; `git push` sends commits and, through
# .githooks/pre-push, the bytes this box made.
#
# Anything after the script name goes to the pull, so `bash data/setup.sh --view train_Z5`
# takes only what one view needs instead of all 423,290 images.
set -e
cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet HEAD; then
    echo "setup.sh: tracked files in this checkout are modified, so nothing was pulled." >&2
    echo "  Commit or stash them (git status) and rerun." >&2
    exit 1
fi

# This script's own pull, below, is the only OSS pull this run does. On a rerun
# core.hooksPath is already set, so the `git pull` here fires post-merge, which would
# pull again; PII_SETUP tells that hook to stand down.
export PII_SETUP=1
if ! git pull --ff-only; then
    echo "setup.sh: git pull --ff-only failed, so nothing was fetched from OSS." >&2
    echo "  The branch has diverged from its upstream or has no upstream at all." >&2
    echo "  Sort the git state out (git status; git log --oneline --graph HEAD @{u}) and rerun." >&2
    exit 1
fi
unset PII_SETUP

git config core.hooksPath .githooks
echo "core.hooksPath = $(git config --get core.hooksPath)"
python3 data/oss_sync.py pull "$@"
