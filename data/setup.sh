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
head_before=$(git rev-parse HEAD)
export PII_SETUP=1
if ! git pull --ff-only; then
    echo "setup.sh: git pull --ff-only failed, so nothing was fetched from OSS." >&2
    echo "  The branch has diverged from its upstream or has no upstream at all." >&2
    echo "  Sort the git state out (git status; git log --oneline --graph HEAD @{u}) and rerun." >&2
    exit 1
fi
unset PII_SETUP

# The pull can replace THIS FILE while the shell is still reading it, and a shell reads a
# script by byte offset, so the rest of the run would be whatever happens to sit at that
# offset in the new text. When the pull moved setup.sh, hand over to the new one.
if [ "$(git rev-parse HEAD)" != "$head_before" ] \
   && ! git diff --quiet "$head_before" HEAD -- data/setup.sh; then
    echo "setup.sh: the pull updated this script; running the new one."
    exec sh data/setup.sh "$@"
fi

git config core.hooksPath .githooks
echo "core.hooksPath = $(git config --get core.hooksPath)"
# `set -e` would abort here, but the pull's exit code is the script's result and the
# render below has to run either way, so the failure is carried rather than raised.
set +e
python3 data/oss_sync.py pull "$@"
pull_rc=$?
set -e

# The view manifests (views/<name>/{scrfd.txt,d2.json,summary.json}) are rendered, not
# tracked and not on OSS, and the trainer configs name them by path, so a fresh clone
# cannot start a run without this. It reads only the indexes, so it is correct after a
# subset pull too, and it is cheap (about a minute for all of them).
python3 data/build_view.py render --all

# The pull's own result is what this script reports: a view row whose bytes are not on
# OSS yet is a real failure and must not be hidden by the render succeeding.
exit $pull_rc
