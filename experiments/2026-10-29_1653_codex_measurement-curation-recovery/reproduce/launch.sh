#!/bin/sh
set -eu
# This family had separately frozen acquisition, curation and admission invocations.
# Print the selected exact receipt; its argv and source hashes define that stage.
# Use a fresh experiment/volume revision when re-executing; preserve prior artifacts.
base=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "${1:-index}" in
  index) cat "$base/launch.json" ;;
  curation) cat "$base/curation-launch.json" ;;
  recovery-admission) cat "$base/recovery-admission-launch.json" ;;
  *) printf '%s\n' 'Usage: launch.sh [index|curation|recovery-admission]' >&2; exit 2 ;;
esac
