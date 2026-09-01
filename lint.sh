#!/usr/bin/env bash
set -euo pipefail

# `ruff check .`, not a hand-written directory list.
#
# The list was the bug: `scripts/` was missing from it, so the largest
# directory in the repo was invisible to CI and hid 383 findings. Appending
# `scripts/` would have fixed that instance and left the mechanism -- the next
# top-level package added would be silently unlinted the same way.
#
# What must NOT be linted is declared in `ruff.toml` under `extend-exclude`,
# so every caller of ruff honours it, not just this script.
ruff check .

# The formatter is policy, not documentation.
#
# `ruff.toml` grew a `[format]` stanza while only the files that happened to
# contain a long line were ever formatted -- so the tree was half managed and
# nothing detected the split. The remaining files are formatted now, and this
# keeps them that way: without it, the next `ruff format` on an untouched file
# drags an unrelated reformat into someone's diff.
ruff format --check .
