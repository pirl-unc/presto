"""Tests for the dual-corpus detectability transfer set.

Detectability is trained and scored on shotgun rows but used on MHC rows.
Peptides in both corpora are the only in-domain test of whether it transfers.
"""

from presto.data.bulk_ms import BulkMSRecord, dual_corpus_transfer_set


def _bulk(peptides):
    return [BulkMSRecord(peptide=p, machinery="trypsin", detectability_label=1.0) for p in peptides]


def test_selects_only_peptides_present_in_both_corpora():
    bulk = _bulk(["SIINFEKL", "AAAAAAAA", "GILGFVFTL"])
    overlap, stats = dual_corpus_transfer_set(["SIINFEKL", "GILGFVFTL", "NOPE"], bulk)
    assert {r.peptide for r in overlap} == {"SIINFEKL", "GILGFVFTL"}
    assert stats["n_overlap_peptides"] == 2


def test_matching_is_case_insensitive():
    overlap, _ = dual_corpus_transfer_set(["siinfekl"], _bulk(["SIINFEKL"]))
    assert len(overlap) == 1


def test_empty_overlap_is_reported_not_crashed():
    overlap, stats = dual_corpus_transfer_set(["AAAAAAAA"], _bulk(["CCCCCCCC"]))
    assert overlap == []
    assert stats["n_overlap_records"] == 0
    assert stats["length_histogram"] == {}


def test_length_histogram_supports_reporting_by_mhc_length():
    bulk = _bulk(["SIINFEKL", "SIINFEKLA", "SIINFEKLAA"])
    _, stats = dual_corpus_transfer_set(["SIINFEKL", "SIINFEKLA", "SIINFEKLAA"], bulk)
    assert stats["length_histogram"] == {8: 1, 9: 1, 10: 1}
