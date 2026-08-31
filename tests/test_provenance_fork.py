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
            net.excision_head.length_preference.weight.fill_(1.0)
            protein = _run(net, source="protein")["excision_length_score"].item()
            mhc = _run(net, source="mhc", machinery="proteasome")["excision_length_score"].item()
        assert protein == pytest.approx(1.0), "digest branch should read the table"
        assert mhc == pytest.approx(0.0), "in-vivo branch must contribute no length term"

    def test_missed_cleavage_term_is_digest_only(self, model):
        """The proteasome is processive; internal sites say nothing about it."""
        with torch.no_grad():
            protein = _run(model, source="protein")["excision_missed_cleavage_score"].item()
            mhc = _run(model, source="mhc", machinery="proteasome")["excision_missed_cleavage_score"].item()
        assert mhc == pytest.approx(0.0)
        assert protein < 0.0

    def test_digest_rule_still_applies_on_the_protein_branch(self, model):
        with torch.no_grad():
            k_term = _run(model, peptide="SIINFEAAK", source="protein")["excision_c_terminus_score"].item()
            a_term = _run(model, peptide="SIINFEAAA", source="protein")["excision_c_terminus_score"].item()
        assert k_term > a_term


class TestInVivoGradient:
    """Cellular state trains the in-vivo path, by sweeping rather than indexing.

    History matters here, because this class has been wrong twice.

    First it constructed an `mhc`-source row carrying an excision target by
    hand -- a combination the pipeline never produces, since excision labels
    come only from `data/bulk_ms.py` whose rows are all
    `peptide_source="protein"` -- and was cited as evidence the in-vivo path
    was supervised. It was not.

    Then conditions were routed into the processing *latent*, which did
    supervise them, but by feeding the observed APM state and stimulus into
    the trunk. `docs/assay_modeling_contract.md` names "stimulation context"
    as a forbidden input, so that bought supervision at the cost of a model
    that could not predict presentation without being told the cell's state.

    Now the excision head *sweeps* those axes: `excision_panel_apm` and
    `excision_panel_stimulus` give one predicted logit per condition, and the
    observed condition selects which column the elution loss reads. The
    consequence for these tests is that gradient reaches the whole profile
    table, not only the observed row -- asserting per-row gradient would now
    be asserting the old design.
    """

    def test_the_profiles_are_reachable_from_the_excision_logit(self):
        """Structural check: the in-vivo branch is still wired to an output."""
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        out = _run(net, source="mhc", apm="n_term_trimming",
                   stimulus="ifn_gamma", machinery="proteasome")
        F.binary_cross_entropy_with_logits(
            out["excision_logit"], torch.ones(1)
        ).backward()

        head = net.excision_head
        for name in ("invivo_profile_n", "invivo_profile_c"):
            grad = getattr(head, name).grad
            assert grad is not None and grad.abs().sum().item() > 0, (
                f"{name} is unreachable from the excision logit"
            )

    def test_panels_predict_every_condition(self):
        """One output track per cellular condition, from sequence alone."""
        from presto.data.vocab import APM_PERTURBATIONS, PROCESSING_STIMULI

        net = Presto(d_model=32, n_layers=2, n_heads=4)
        net.eval()
        with torch.no_grad():
            out = _run(net, source="mhc", apm="none", stimulus="none",
                       machinery="proteasome")
        assert out["excision_panel_apm"].shape[-1] == len(APM_PERTURBATIONS)
        assert out["excision_panel_stimulus"].shape[-1] == len(PROCESSING_STIMULI)

    def test_real_pipeline_supervises_cellular_state_via_elution(self):
        """Gap 2, closed and pinned -- now through the panel.

        MHC rows, elution labels only, **no excision label anywhere**. The
        in-vivo profiles must still receive gradient: that was gap 2's whole
        content, and it has to survive the move from conditioning to sweeping.
        """
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset
        from presto.scripts.train_synthetic import compute_loss

        mhc_seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        dataset = PrestoDataset(
            elution_records=[
                ElutionRecord(peptide="LLDGTATLRF", alleles=["HLA-A*02:01"],
                              detected=True, stimulus="ifn_gamma",
                              apm_perturbation="n_term_trimming"),
                ElutionRecord(peptide="SIINFEKLAA", alleles=["HLA-A*02:01"],
                              detected=True, stimulus="none",
                              apm_perturbation="none"),
            ],
            mhc_sequences={"HLA-A*02:01": mhc_seq},
            strict_mhc_resolution=False,
        )
        batch = PrestoCollator()([dataset[i] for i in range(len(dataset))])
        assert "excision" not in batch.target_masks, (
            "fixture must carry no excision label"
        )

        net = Presto(d_model=32, n_layers=2, n_heads=4)
        loss, _, _ = compute_loss(net, batch, "cpu")
        loss.backward()

        head = net.excision_head
        for name in (
            "invivo_profile_c",
            "invivo_profile_n",
            "stimulus_profile_c",
            "invivo_bias",
        ):
            grad = getattr(head, name).grad
            assert grad is not None and grad.abs().sum().item() > 0, (
                f"{name} received no gradient from elution labels alone; "
                "gap 2 has reopened"
            )

    def test_the_trunk_no_longer_takes_cellular_state(self):
        """The input path that the previous fix relied on is gone."""
        net = Presto(d_model=32, n_layers=2, n_heads=4)
        assert not hasattr(net, "processing_condition_embed"), (
            "cellular state is being fed into the trunk again"
        )

    def test_mil_path_carries_per_instance_state(self):
        """The elution loss runs through the bag path whenever MIL is active.

        That forward is separate, and while it omitted provenance every
        instance collapsed to the default condition -- so the whole Tier 3
        signal landed on one embedding row regardless of the sample's state.
        """
        from presto.data.collate import PrestoCollator
        from presto.data.loaders import ElutionRecord, PrestoDataset

        mhc_seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        dataset = PrestoDataset(
            elution_records=[
                ElutionRecord(peptide="LLDGTATLRF", alleles=["HLA-A*02:01"],
                              detected=True, stimulus="ifn_gamma",
                              apm_perturbation="n_term_trimming"),
                ElutionRecord(peptide="SIINFEKLAA", alleles=["HLA-A*02:01"],
                              detected=True, stimulus="none",
                              apm_perturbation="none"),
            ],
            mhc_sequences={"HLA-A*02:01": mhc_seq},
            strict_mhc_resolution=False,
        )
        batch = PrestoCollator()([dataset[i] for i in range(len(dataset))])
        assert batch.mil_bag_label is not None, "fixture should exercise the MIL path"
        apm = batch.mil_provenance["apm_perturbation_idx"].tolist()
        assert len(set(apm)) > 1, "MIL instances collapsed to one condition"

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

        This assertion previously read "unperturbed cells carry basal
        interferon tone" -- the overclaim the basal -> none rename exists to
        retire. Resting cells do carry tonic signaling, but the corpus rarely
        records whether a sample was resting at all, so the token asserts only
        the absence of a recorded treatment.
        """
        assert stimulus_for_condition(None) == "none"
        assert PROCESSING_STIMULUS_TO_IDX["none"] == 0
