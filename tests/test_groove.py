"""The groove is whatever mhcseqs says it is.

presto used to carry its own cysteine-anchored domain parser. Measured against
mhcseqs on 1,500 alleles from `data/mhc_index.csv`, the two agreed on 19.5% of
them. Class I was consistently off by one -- presto kept the leader's terminal
alanine, yielding `AGSHSMRY...` where the mature chain is `GSHSMRY...` -- and
class II differed by nine residues or more. For a model that reads groove
positions as pocket-forming residues, position `k` did not mean what it was
supposed to mean.

presto also parsed 100% of the index, which sounds like coverage and was not:
the 0.9% mhcseqs declines are TAP2 transporters, HLA-DM chaperones and null
alleles. Those proteins have no peptide groove, and truncating them to the
right length does not give them one.

So these tests pin two things: that presto reports mhcseqs' answer unchanged,
and that a protein with no groove yields no groove.
"""

import csv
import random
from pathlib import Path

import pytest

from presto.data.groove import PreparedMHCInput, groove_record, prepare_mhc_input

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Full-length class I heavy chains. The leader is the canonical 24-mer for
#: both, so the mature chain -- and therefore the first groove half -- must
#: begin at index 24.
CLASS_I = {
    "HLA-A*02:01": (
        "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYW"
        "DGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTK"
        "HKWEAAHVAEQLRAYLDGTCVEWLRRYLENGKETLQRT"
    ),
    "HLA-B*07:02": (
        "MLVMAPRTVLLLLSAALALTETWAGSHSMRYFYTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPREEPRAPWIEQEGPEYW"
        "DRNTQIYKAQAQTDRESLRNLRGYYNQSEAGSHTLQSMYGCDVGPDGRLLRGHDQYAYDGKDYIALNEDLRSWTAADTAAQITQ"
        "RKWEAAREAEQRRAYLEGECVEWLRRYLENGKDKLERA"
    ),
}
CLASS_I_LEADER_LEN = 24


class TestTheMatureChainStartsWhereBiologyStartsIt:
    """The signal peptide is not part of the groove."""

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_the_first_groove_half_begins_at_the_mature_start(self, allele):
        seq = CLASS_I[allele]
        prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
        assert prepared.groove_half_1.startswith(seq[CLASS_I_LEADER_LEN:][:20])

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_the_leader_terminal_residue_is_excluded(self, allele):
        """The exact off-by-one that was there before.

        Residue 24 is the leader's final alanine. Including it produced
        `AGSHSMRY...`, one residue upstream of the real N-terminus, on every
        class I allele.
        """
        seq = CLASS_I[allele]
        assert seq[CLASS_I_LEADER_LEN - 1] == "A", "fixture assumption"
        groove = prepare_mhc_input(mhc_a=seq, mhc_class="I").groove_half_1
        assert not groove.startswith("A" + seq[CLASS_I_LEADER_LEN : CLASS_I_LEADER_LEN + 8])
        assert groove.startswith("GSHSMRY"), "conserved class I mature N-terminus"

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_both_halves_are_substrings_of_the_input(self, allele):
        seq = CLASS_I[allele]
        prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
        assert prepared.groove_half_1 in seq
        assert prepared.groove_half_2 in seq

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_the_halves_do_not_overlap_and_are_ordered(self, allele):
        seq = CLASS_I[allele]
        prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
        start1 = seq.index(prepared.groove_half_1)
        start2 = seq.index(prepared.groove_half_2)
        assert start1 < start2, "alpha1 precedes alpha2"
        assert start1 + len(prepared.groove_half_1) <= start2


class TestAProteinWithNoGrooveGetsNoGroove:
    """presto's old parser gave TAP2 and HLA-DM a groove. They do not have one.

    Truncating an arbitrary protein to roughly the right length is not
    extraction, and a downstream model cannot tell the difference between that
    and a real binding cleft.
    """

    def test_a_peptide_transporter_is_refused(self):
        """TAP2 moves peptides across the ER membrane; it presents nothing."""
        record = _index_sequence_for(lambda row: row["gene"] == "TAP2")
        if record is None:
            pytest.skip("no TAP2 row in the shipped index")
        prepared = prepare_mhc_input(mhc_a=record, mhc_class="I")
        assert prepared.groove_half_1 == ""
        assert prepared.used_fallback is True

    def test_a_refusal_is_reported_not_silently_padded(self):
        prepared = prepare_mhc_input(mhc_a="MMMMMMMMMM", mhc_class="I")
        assert prepared.groove_half_1 == ""
        assert prepared.groove_half_2 == ""
        assert prepared.used_fallback is True
        assert prepared.groove_status_a and prepared.groove_status_a != "ok"

    def test_strict_callers_get_an_exception(self):
        with pytest.raises(ValueError, match="Class-I groove extraction failed"):
            prepare_mhc_input(mhc_a="MMMMMMMMMM", mhc_class="I", allow_fallback_truncation=False)

    def test_an_empty_sequence_is_not_a_groove(self):
        prepared = prepare_mhc_input(mhc_a="", mhc_class="I")
        assert prepared.groove_half_1 == ""
        assert prepared.used_fallback is True


class TestPrestoReportsMhcseqsUnchanged:
    """No second opinion. If these drift apart, presto has grown a parser again."""

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_prepare_matches_the_underlying_record(self, allele):
        seq = CLASS_I[allele]
        record = groove_record(seq, mhc_class="I", chain="alpha", allele=allele)
        prepared = prepare_mhc_input(mhc_a=seq, mhc_class="I")
        assert prepared.groove_half_1 == record.groove_half_1
        assert prepared.groove_half_2 == record.groove_half_2

    @pytest.mark.parametrize("allele", sorted(CLASS_I))
    def test_the_record_matches_mhcseqs_itself(self, allele):
        mhcseqs = pytest.importorskip("mhcseqs")
        seq = CLASS_I[allele]
        upstream = mhcseqs.extract_groove(seq, mhc_class="I", chain="alpha")
        record = groove_record(seq, mhc_class="I", chain="alpha", allele=allele)
        assert record.groove_half_1 == upstream.groove1
        assert record.groove_half_2 == upstream.groove2
        assert record.mature_start == upstream.mature_start

    def test_the_homegrown_parsers_are_gone(self):
        """A regression guard with teeth: the names must not come back.

        Re-adding a local parser is how the two answers diverged in the first
        place, and it diverged silently -- both produced plausible sequences.
        """
        import presto.data.groove as groove

        for name in (
            "parse_class_i",
            "parse_class_ii_alpha",
            "parse_class_ii_beta",
            "extract_groove",
            "find_cys_pairs",
            "classify_cys_pair",
        ):
            assert not hasattr(groove, name), (
                f"{name} is back in presto.data.groove; the groove must come "
                f"from mhcseqs, not from a second implementation here"
            )


class TestClassIIUsesBothChains:
    """Alpha supplies the first groove half, beta the second."""

    @staticmethod
    def _pair():
        mhcseqs = pytest.importorskip("mhcseqs")
        alpha = mhcseqs.lookup("HLA-DRA*01:01")
        beta = mhcseqs.lookup("HLA-DRB1*01:01")
        if not (alpha.ok and beta.ok):
            pytest.skip("mhcseqs cannot resolve the DR reference pair")
        return alpha.sequence, beta.sequence

    def test_each_half_comes_from_its_own_chain(self):
        alpha_seq, beta_seq = self._pair()
        prepared = prepare_mhc_input(mhc_a=alpha_seq, mhc_b=beta_seq, mhc_class="II")
        assert prepared.groove_half_1 in alpha_seq
        assert prepared.groove_half_2 in beta_seq
        assert prepared.used_fallback is False

    def test_a_missing_beta_chain_leaves_the_second_half_empty(self):
        alpha_seq, _ = self._pair()
        prepared = prepare_mhc_input(mhc_a=alpha_seq, mhc_b="", mhc_class="II")
        assert prepared.groove_half_1
        assert prepared.groove_half_2 == ""
        assert prepared.used_fallback is True


class TestTheIndexParsesConsistently:
    """Sampled across the shipped index, not just the two hand-picked alleles."""

    @staticmethod
    def _sample(n=200, seed=0):
        path = REPO_ROOT / "data" / "mhc_index.csv"
        if not path.is_file():
            pytest.skip("mhc_index.csv is not present")
        rows = []
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("sequence") or "").strip():
                    rows.append(row)
        random.Random(seed).shuffle(rows)
        return rows[:n]

    def test_every_reported_half_is_a_substring_of_its_source(self):
        """The one invariant that must never break: we slice, we do not invent."""
        offenders = []
        for row in self._sample():
            seq = row["sequence"].strip().upper()
            chain = None
            if row["mhc_class"] == "II":
                chain = (
                    "alpha"
                    if any(g in (row["gene"] or "") for g in ("DRA", "DQA", "DPA", "DMA", "DOA"))
                    else "beta"
                )
            record = groove_record(
                seq, mhc_class=row["mhc_class"], chain=chain, allele=row["normalized"]
            )
            for half in (record.groove_half_1, record.groove_half_2):
                if half and half not in seq:
                    offenders.append(row["normalized"])
        assert offenders == [], f"fabricated groove sequence for: {offenders[:5]}"

    def test_class_i_grooves_never_retain_the_leader(self):
        """A class I groove starting one residue early is the old bug returning."""
        offenders = []
        for row in self._sample():
            if row["mhc_class"] != "I":
                continue
            seq = row["sequence"].strip().upper()
            record = groove_record(seq, mhc_class="I", chain="alpha", allele=row["normalized"])
            if not record.ok or not record.groove_half_1:
                continue
            start = seq.index(record.groove_half_1)
            if start != record.mature_start:
                offenders.append((row["normalized"], start, record.mature_start))
        assert offenders == [], (
            "the first groove half must begin exactly at the mature start; "
            f"offenders: {offenders[:5]}"
        )


class TestPreparedInputShapeIsStable:
    """Callers outside this module depend on the dataclass, not the internals."""

    def test_fields_survive_the_rewrite(self):
        prepared = prepare_mhc_input(mhc_a=CLASS_I["HLA-A*02:01"], mhc_class="I")
        assert isinstance(prepared, PreparedMHCInput)
        for field in (
            "mhc_class",
            "groove_half_1",
            "groove_half_2",
            "groove_status_a",
            "groove_status_b",
            "used_fallback",
        ):
            assert hasattr(prepared, field)

    def test_an_unknown_class_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported MHC class"):
            prepare_mhc_input(mhc_a=CLASS_I["HLA-A*02:01"], mhc_class="bogus")

    def test_class_iii_is_not_class_ii(self):
        """MHC class III is complement and TNF -- it has no peptide groove.

        `normalize_mhc_class` prefix-matched Roman numerals, so "III" became
        "II" and the chain was handed to a class II groove parser. presto#42.
        """
        with pytest.raises(ValueError, match="Unsupported MHC class"):
            prepare_mhc_input(mhc_a=CLASS_I["HLA-A*02:01"], mhc_class="III")


def _index_sequence_for(predicate):
    path = REPO_ROOT / "data" / "mhc_index.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seq = (row.get("sequence") or "").strip().upper()
            if seq and predicate(row):
                return seq
    return None
