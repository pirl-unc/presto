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
    PROCESSING_INDUCER_TO_IDX,
    apm_group_for_genes,
    inducer_for_condition,
)
from presto.models.presto import Presto


def _encode(seq):
    return torch.tensor([[AA_TO_IDX[c] for c in seq]], dtype=torch.long)


def _provenance(source, apm="none", inducer="basal"):
    return {
        "peptide_source_idx": torch.tensor([PEPTIDE_SOURCE_TO_IDX[source]]),
        "apm_perturbation_idx": torch.tensor([APM_PERTURBATION_TO_IDX[apm]]),
        "processing_inducer_idx": torch.tensor([PROCESSING_INDUCER_TO_IDX[inducer]]),
    }


def _run(model, peptide="SIINKEKAK", source="protein", apm="none",
         inducer="basal", machinery="trypsin"):
    return model(
        pep_tok=_encode(peptide),
        mhc_a_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(0)),
        mhc_b_tok=torch.randint(4, 24, (1, 40), generator=torch.Generator().manual_seed(1)),
        flank_n_tok=_encode("GGGGR"),
        flank_c_tok=_encode("AAAAA"),
        mhc_class="I",
        machinery=[machinery],
        provenance=_provenance(source, apm, inducer),
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
        """
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        net.eval()
        with torch.no_grad():
            protein = _run(net, source="protein")["excision_s_len"].item()
            mhc = _run(net, source="mhc", machinery="proteasome")["excision_s_len"].item()
        assert mhc == pytest.approx(0.0)
        assert protein != pytest.approx(0.0)

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
    def test_gradient_reaches_the_in_vivo_path(self):
        """This is the defect: no row with machinery=proteasome carried an
        excision label, so the beta1/beta2/beta5 mixture sat at its init."""
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        out = _run(net, source="mhc", apm="n_term_trimming",
                   inducer="ifn_gamma", machinery="proteasome")
        F.binary_cross_entropy_with_logits(out["excision_logit"], torch.ones(1)).backward()

        head = net.excision_head
        erap = APM_PERTURBATION_TO_IDX["n_term_trimming"]
        assert head.invivo_profile_n.grad[erap].abs().sum().item() > 0
        assert head.invivo_profile_c.grad[erap].abs().sum().item() > 0
        assert head.inducer_profile_c.grad[
            PROCESSING_INDUCER_TO_IDX["ifn_gamma"]
        ].abs().sum().item() > 0

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
        ("IFN_alpha_treatment", "ifn_ab"),
        ("TNF_alpha_treatment", "tnf_alpha"),
        ("unperturbed", "basal"),
        ("", "basal"),
    ])
    def test_conditions_map_to_inducers(self, condition, expected):
        assert inducer_for_condition(condition) == expected

    def test_unperturbed_defaults_to_basal_not_zero(self):
        """Unperturbed cells carry basal interferon tone, not none."""
        assert inducer_for_condition(None) == "basal"
        assert PROCESSING_INDUCER_TO_IDX["basal"] == 0
