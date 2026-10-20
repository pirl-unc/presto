"""Fraction/replicate counts alone cannot establish detectability labels (#70)."""

import pytest

from presto.data.bulk_ms import records_from_bulk_frame


@pytest.mark.parametrize("depth", [None, 12, 14, 39, 46, 50, 70])
@pytest.mark.parametrize("replicates", [None, 1, 3])
def test_unqualified_bulk_rows_never_acquire_detectability_targets(depth, replicates):
    records, stats = records_from_bulk_frame(
        [
            {
                "peptide": "ACDEFGHIK",
                "digestion_enzyme": "Trypsin/P",
                "n_fractions_in_run": depth,
                "n_replicates_detected": replicates,
            }
        ],
        excision_negative_ratio=1,
    )
    assert records
    assert all(record.detectability_label is None for record in records)
    assert records[0].excision_label == 1.0
    assert records[1].excision_label == 0.0
    assert records[0].n_replicates_detected == replicates
    assert stats["n_detectability_labels"] == 0
    assert stats["detectability_status"] == "unavailable_without_verified_search_scope"


def test_unqualified_bulk_labels_are_masked_in_real_collation():
    from presto.data.collate import PrestoCollator
    from presto.data.loaders import PrestoDataset

    records, _ = records_from_bulk_frame(
        [{"peptide": "ACDEFGHIK", "digestion_enzyme": "Trypsin/P"}]
    )
    samples = PrestoDataset(bulk_ms_records=records).samples
    batch = PrestoCollator()(samples)
    assert "ms_detectability" not in batch.target_masks
    assert batch.target_masks["excision"].all()
    assert all(
        sample.target_provenance.get("ms_detectability") != "bulk_depth_proxy" for sample in samples
    )
