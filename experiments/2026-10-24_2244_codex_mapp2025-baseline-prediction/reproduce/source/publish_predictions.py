"""Lossless compact publication of the canonical per-example CSV dump."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results/predictions.csv.gz"
output = ROOT / "results/predictions.parquet"
if output.exists():
    raise FileExistsError(output)
frame = pd.read_csv(source, float_precision="round_trip")
table = pa.Table.from_pandas(frame, preserve_index=False)
pq.write_table(table, output, compression="zstd")
assert pq.read_table(output).equals(table)
receipt = {
    "n_prediction_rows": len(frame),
    "conversion": "CSV round-trip float parser -> Arrow -> Parquet -> exact Arrow equality",
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "published_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "source_bytes": source.stat().st_size,
    "published_bytes": output.stat().st_size,
}
(ROOT / "results/prediction-publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
