"""Tests for the peptide-source fork and cellular-state conditioning.

The single flat `machinery` axis conflated sample prep with cellular state.
Because MHC ligands are never digested and shotgun proteins are extracted
whole, that axis was perfectly anti-correlated with the corpus -- a pure corpus
indicator, and the reason the in-vivo half of the head never received gradient.
"""

import pytest
import torch
import torch.nn.functional as F

from presto.data.vocab import (
    AA_TO_IDX,
    APM_PERTURBATION_TO_IDX,
    PEPTIDE_SOURCE_TO_IDX,
    PROCESSING_STIMULUS_TO_IDX,
    apm_group_for_genes,
    stimulus_for_condition,
)
from presto.models.presto import Presto


def _encode(seq):
    return torch.tensor([[AA_TO_IDX[c] for c in seq]], dtype=torch.long)


def _provenance(source, apm="none", stimulus="none"):
    return {
        "peptide_source_idx": torch.tensor([PEPTIDE_SOURCE_TO_IDX[source]]),
        "apm_perturbation_idx": torch.tensor([APM_PERTURBATION_TO_IDX[apm]]),
        "processing_stimulus_idx": torch.tensor([PROCESSING_STIMULUS_TO_IDX[stimulus]]),
    }


def _run(model, peptide="SIINKEKAK", source="protein", apm="none",
         stimulus="none", machinery="trypsin"):
    return model(
        pep_tok=_encode(peptide),
        mhc_a_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(0)),
        mhc_b_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(1)),
        flank_n_tok=_encode("GGGGR"),
        flank_c_tok=_encode("AAAAA"),
        mhc_class="I",
        machinery=[machinery],
        provenance=_provenance(source, apm, stimulus),
    )


@pytest.fixture
def model():
    net = Presto(d_model=32, n_layers=2, n_heads=4)
    net.eval()
    return net


class TestSourceFork:
    def test_length_term_is_digest_only(self):
        """Class I length is the MHC groove and TAP, not the protease.

        Keeping a length term on the in-vivo branch would credit the protease
        for MHC selection.

        The length table is deliberately zero-initialized, so at init both
        branches read 0 and the gate is untestable. Give it a non-zero value
        first, then check that only the protein branch sees it -- otherwise
        this passes for the wrong reason (it previously did, because a blanket
        embedding re-init was randomizing the table).
        """
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        net.eval()
        with torch.no_grad():
            net.excision_head.length_score.weight.fill_(1.0)
            protein = _run(net, source="protein")["excision_s_len"].item()
            mhc = _run(net, source="mhc", machinery="proteasome")["excision_s_len"].item()
        assert protein == pytest.approx(1.0), "digest branch should read the table"
        assert mhc == pytest.approx(0.0), "in-vivo branch must contribute no length term"

    def test_missed_cleavage_term_is_digest_only(self, model):
        """The proteasome is processive; internal sites say nothing about it."""
        with torch.no_grad():
            protein = _run(model, source="protein")["excision_s_internal"].item()
            mhc = _run(model, source="mhc", machinery="proteasome")["excision_s_internal"].item()
        assert mhc == pytest.approx(0.0)
        assert protein < 0.0

    def test_digest_rule_still_applies_on_the_protein_branch(self, model):
        with torch.no_grad():
            k_term = _run(model, peptide="SIINFEAAK", source="protein")["excision_s_c"].item()
            a_term = _run(model, peptide="SIINFEAAA", source="protein")["excision_s_c"].item()
        assert k_term > a_term


class TestInVivoGradient:
    """Scope warning, learned the hard way.

    These construct an `mhc`-source row carrying an excision target by hand.
    **The data pipeline never produces that combination**: excision labels come
    only from `data/bulk_ms.py`, whose rows are all `peptide_source="protein"`.
    So these prove the wiring *can* carry gradient, not that any real data does
    -- and an earlier version of this file was cited as evidence that the
    in-vivo path was supervised, which it is not.

    `test_real_pipeline_still_starves_the_in_vivo_path` below pins the actual
    state of affairs. Delete it only when in-vivo excision labels exist.
    """

    def test_wiring_can_carry_gradient_to_the_in_vivo_path(self):
        """Structural check only -- see the class docstring."""
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        out = _run(net, source="mhc", apm="n_term_trimming",
                   stimulus="ifn_gamma", machinery="proteasome")
        F.binary_cross_entropy_with_logits(out["excision_logit"], torch.ones(1)).backward()

        head = net.excision_head
        erap = APM_PERTURBATION_TO_IDX["n_term_trimming"]
        assert head.invivo_profile_n.grad[erap].abs().sum().item() > 0
        assert head.invivo_profile_c.grad[erap].abs().sum().item() > 0
        assert head.stimulus_profile_c.grad[
            PROCESSING_STIMULUS_TO_IDX["ifn_gamma"]
        ].abs().sum().item() > 0

    def test_real_pipeline_still_starves_the_in_vivo_path(self):
        """The in-vivo tables receive exactly zero gradient from real data.

        Every excision-supervised row is protein-sourced, so `(1 - is_protein)`
        zeroes the in-vivo terms on every row that carries a label. Closing
        this needs excision labels on MHC rows, which needs a source of in-vivo
        excision negatives -- the same unsolved problem as detectability
        negatives. Until then ~482 parameters train to nothing.
        """
        from presto.data.bulk_ms import BulkMSRecord
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset
        from presto.scripts.train_synthetic import compute_loss

        mhc_seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        dataset = PrestoDataset(
            bulk_ms_records=[
                BulkMSRecord(peptide="SIINFEKLK", machinery="trypsin",
                             detectability_label=1.0, excision_label=1.0),
            ],
            elution_records=[
                ElutionRecord(peptide="LLDGTATLRF", alleles=["HLA-A*02:01"],
                              detected=True, stimulus="ifn_gamma",
                              apm_perturbation="n_term_trimming"),
            ],
            mhc_sequences={"HLA-A*02:01": mhc_seq},
            strict_mhc_resolution=False,
        )
        batch = PrestoCollator()([dataset[i] for i in range(len(dataset))])
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        loss, _, _ = compute_loss(net, batch, "cpu")
        loss.backward()

        head = net.excision_head
        for name in ("invivo_profile_c", "invivo_profile_n",
                     "stimulus_profile_c", "invivo_bias"):
            grad = getattr(head, name).grad
            total = 0.0 if grad is None else grad.abs().sum().item()
            assert total == 0.0, (
                f"{name} now receives gradient from the real pipeline -- if in-vivo "
                "excision labels were added, delete this test and update "
                "docs/model_io_contract.md, which records this as an open gap"
            )

    def test_only_the_present_condition_receives_gradient(self):
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        out = _run(net, source="mhc", apm="n_term_trimming", machinery="proteasome")
        F.binary_cross_entropy_with_logits(out["excision_logit"], torch.ones(1)).backward()
        absent = APM_PERTURBATION_TO_IDX["mhc_null"]
        assert net.excision_head.invivo_profile_n.grad[absent].abs().sum().item() == 0.0


class TestConditionMapping:
    @pytest.mark.parametrize("genes,expected", [
        ("erap1", "n_term_trimming"),
        ("erap1;erap2", "n_term_trimming"),
        ("tap1", "peptide_supply"),
        ("b2m", "mhc_null"),
        ("tapbp;calr", "loading_complex"),
        ("hla_dm", "class_ii_loading"),
        ("", "none"),
        (None, "none"),
    ])
    def test_genes_map_to_mechanism_groups(self, genes, expected):
        assert apm_group_for_genes(genes) == expected

    def test_severity_ordering_when_several_are_perturbed(self):
        """B2M loss abolishes class I, so it dominates whatever else is out."""
        assert apm_group_for_genes("erap1;b2m;tap1") == "mhc_null"
        assert apm_group_for_genes("erap1;tap1") == "peptide_supply"

    @pytest.mark.parametrize("condition,expected", [
        ("IFN_gamma_treatment", "ifn_gamma"),
        ("IFN_alpha_treatment", "ifn_type1"),
        ("TNF_alpha_treatment", "tnf_alpha"),
        ("unperturbed", "none"),
        ("", "none"),
    ])
    def test_conditions_map_to_stimuli(self, condition, expected):
        assert stimulus_for_condition(condition) == expected

    def test_unrecorded_condition_defaults_to_none(self):
        """`none` is a catch-all, not a claim about interferon tone.

        This previously asserted "unperturbed cells carry basal interferon
        tone" -- the overclaim the basal -> none rename exists to retire.
        Resting cells do carry tonic signalling, but the corpus rarely records
        whether a sample was resting at all, so the token asserts only the
        absence of a recorded treatment.
        """
        assert stimulus_for_condition(None) == "none"
        assert PROCESSING_STIMULUS_TO_IDX["none"] == 0
