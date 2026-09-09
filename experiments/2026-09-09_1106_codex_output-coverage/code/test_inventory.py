"""Evidence counts reconcile repeated/null rows without claiming other modalities are MS."""

import csv
import importlib.util
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "coverage_inventory", Path(__file__).with_name("launch.py")
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def database():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE excluded (source_file TEXT, pmid TEXT, peptide TEXT, kind TEXT)")
    return db


def test_parquet_count_preserves_duplicates_and_null_pmid_is_not_a_match(tmp_path):
    path = tmp_path / "observations.parquet"
    pq.write_table(
        pa.table(
            {
                "pmid": [123, 123, 456, None],
                "peptide": ["ACDEFGHIK", "ACDEFGHIK", "LMNPQRSTV", "ACDEFGHIK"],
            }
        ),
        path,
        row_group_size=1,
    )
    with database() as db:
        result = inventory.scan_parquet(path, {"123": {}}, db)
        assert result["scanned_rows"] == result["rows"] == 4
        assert result["row_groups"] == 4
        assert result["rows_from_excluded_ms_studies"] == 2
        assert db.execute("SELECT COUNT(*),COUNT(DISTINCT peptide) FROM excluded").fetchone() == (
            2,
            1,
        )
        assert db.execute("SELECT DISTINCT kind FROM excluded").fetchall() == [("ms",)]


def test_merged_inventory_keeps_binding_and_tcell_distinct_from_ms(tmp_path):
    path = tmp_path / "merged_deduped.tsv"
    rows = [
        dict(pmid="123", peptide="ACDEFGHIK", record_type="binding", source="iedb"),
        dict(pmid="123", peptide="ACDEFGHIK", record_type="tcell", source="iedb"),
        dict(pmid="456", peptide="LMNPQRSTV", record_type="elution", source="cedar"),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with database() as db:
        result = inventory.scan_merged(path, {"123": {}}, db, tmp_path)
        assert result["rows"] == 3
        assert result["rows_from_excluded_ms_studies"] == 2
        assert result["record_types"] == {"binding": 1, "tcell": 1, "elution": 1}
        assert result["sources"] == {"iedb": 2, "cedar": 1}
        assert db.execute("SELECT kind FROM excluded ORDER BY kind").fetchall() == [
            ("binding",),
            ("tcell",),
        ]
    with Path(result["excluded_subset"]).open() as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == rows[:2]
