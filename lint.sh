#!/usr/bin/env bash
set -euo pipefail

ruff check *.py cli/ data/ inference/ models/ training/ tests/
