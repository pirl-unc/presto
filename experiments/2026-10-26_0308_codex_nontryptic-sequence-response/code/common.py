"""Small explicit artifact operations; no biological ingestion or fitting."""

import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def request():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    return read(parser.parse_args().requests)
