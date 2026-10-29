#!/bin/sh
set -eu
python "$(dirname "$0")/../code/benchmark.py" "$@"
