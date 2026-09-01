"""Coverage for the shotgun-MS ingest path.

`records_from_bulk_frame` sat at 35% line coverage while carrying three
changes made in this branch -- the encodable-peptide guard, the dropped-row
counter, and chunked streaming. All three were reasoned about and none were
exercised, which is the same shape as the bugs this branch spent its time
finding.
"""

import pytest

pd = pytest.importorskip("pandas")

from presto.data.bulk_ms import (  # noqa: E402
    _iter_frame_rows,
    records_from_bulk_frame,
)


def _frame(rows):
    return pd.DataFrame(rows)


def _row(peptide="SAMPLERPEP", enzyme="trypsin", **kw):
    row = {
        "peptide": peptide,
        "digestion_enzyme": enzyme,
        "n_fractions_in_run": 3,
        "n_replicates_detected": 2,
        "uniprot_acc": "P12345",
        "cell_line_name": "HeLa",
    }
    row.update(kw)
    return row


class TestPeptideAdmissibility:
    @pytest.mark.parametrize(
        "peptide",
        ["AILEVCGUKL", "ILAETVAXV + OTH(X8)", "SIINFEKL*", ""],
    )
    def test_unencodable_peptides_are_dropped_and_counted(self, peptide):
        """Shotgun peptides come from whole-proteome digests, so genuine but
        unmodelled residues (selenocysteine) do occur."""
        records, stats = records_from_bulk_frame(_frame([_row(peptide=peptide)]))
        assert records == []
        assert stats["n_unencodable_peptides"] == 1

    def test_the_counter_is_reported(self):
        """It was incremented and never surfaced, so a corpus where peptides
        silently vanished reported identical stats to one where they never
        existed."""
        _, stats = records_from_bulk_frame(_frame([_row()]))
        assert "n_unencodable_peptides" in stats

    @pytest.mark.parametrize("peptide", ["SIINFEKLA", "NXVPMVATV", "siinfekla"])
    def test_encodable_peptides_survive(self, peptide):
        records, stats = records_from_bulk_frame(_frame([_row(peptide=peptide)]))
        assert records
        assert stats["n_unencodable_peptides"] == 0

    def test_every_emitted_peptide_tokenizes(self):
        from presto.data.tokenizer import Tokenizer

        frame = _frame(
            [
                _row(peptide="SIINFEKLA"),
                _row(peptide="AILEVCGUKL"),
                _row(peptide="LLDGTATLRF"),
            ]
        )
        records, _ = records_from_bulk_frame(frame)
        tokenizer = Tokenizer()
        for record in records:
            tokenizer.encode(record.peptide)  # must not raise


class TestEnzymeRouting:
    def test_unknown_enzyme_rows_are_skipped(self):
        records, stats = records_from_bulk_frame(_frame([_row(enzyme="some-unlisted-protease")]))
        assert records == []

    @pytest.mark.parametrize("enzyme", ["trypsin", "chymotrypsin", "lysc", "gluc"])
    def test_each_known_enzyme_routes(self, enzyme):
        """One observed record plus its in-silico excision negative."""
        records, stats = records_from_bulk_frame(_frame([_row(enzyme=enzyme)]))
        assert stats["machinery_counts"] == {enzyme: 1}
        assert [r.observed for r in records].count(True) == 1

    @pytest.mark.parametrize("spelling", ["trypsin", "Trypsin", "TRYPSIN"])
    def test_enzyme_lookup_is_case_insensitive(self, spelling):
        """The table happens to match the corpus's capitalization exactly, so
        an exact lookup worked by luck. A lowercase export would have mapped
        every enzyme to `unknown`, and unknown-machinery rows are skipped --
        the whole shotgun corpus would vanish silently."""
        _, stats = records_from_bulk_frame(_frame([_row(enzyme=spelling)]))
        assert stats["machinery_counts"] == {"trypsin": 1}


class TestStreaming:
    def test_uncapped_path_yields_every_row(self):
        frame = _frame([_row(peptide=f"SIINFEKL{c}") for c in "ACDEFG"])
        records, stats = records_from_bulk_frame(frame)
        observed = [r for r in records if r.observed]
        assert len(observed) == 6

    def test_capped_path_respects_the_cap(self):
        frame = _frame([_row(peptide=f"SIINFEKL{c}") for c in "ACDEFG"])
        records, _ = records_from_bulk_frame(frame, max_records=2)
        assert len({r.peptide for r in records if r.observed}) <= 2

    def test_iter_frame_rows_crosses_chunk_boundaries(self):
        frame = _frame([{"n": i} for i in range(25)])
        rows = list(_iter_frame_rows(frame, chunk_size=7))
        assert [r["n"] for r in rows] == list(range(25))

    def test_iter_frame_rows_handles_an_empty_frame(self):
        assert list(_iter_frame_rows(_frame([]), chunk_size=4)) == []
