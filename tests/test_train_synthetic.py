"""Tests for synthetic training loss utilities."""

import types

import pytest
import torch
import torch.nn as nn

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.vocab import (
    TCELL_APC_TYPES,
    TCELL_ASSAY_METHODS,
    TCELL_ASSAY_READOUTS,
    TCELL_CULTURE_CONTEXTS,
    TCELL_STIM_CONTEXTS,
)
from presto.scripts.train_synthetic import (
    build_warmup_cosine_scheduler,
    compute_loss,
    train_epoch,
)


class _DummyOutputsModel(nn.Module):
    """Minimal model stub returning predefined outputs for loss tests."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs

    def forward(self, **kwargs):
        device = kwargs["pep_tok"].device
        out = {}
        for key, value in self._outputs.items():
            if isinstance(value, dict):
                out[key] = {
                    nested_key: nested_value.to(device)
                    for nested_key, nested_value in value.items()
                }
            else:
                out[key] = value.to(device)
        return out


def test_compute_loss_includes_cascade_and_chain_assembly_consistency_terms():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="",
        mhc_class="I",
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])
    model = _DummyOutputsModel(
        {
            "processing_logit": torch.tensor([[-2.0]], dtype=torch.float32),
            "binding_logit": torch.tensor([-3.0], dtype=torch.float32),
            "presentation_logit": torch.tensor([[2.5]], dtype=torch.float32),
            "elution_logit": torch.tensor([[1.5]], dtype=torch.float32),
            "ms_logit": torch.tensor([[1.0]], dtype=torch.float32),
            "immunogenicity_logit": torch.tensor([0.0], dtype=torch.float32),
            "tcell_logit": torch.tensor([[0.0]], dtype=torch.float32),
            "assays": {
                "KD_nM": torch.tensor([[3.0]], dtype=torch.float32),
                "IC50_nM": torch.tensor([[5.0]], dtype=torch.float32),
                "EC50_nM": torch.tensor([[1.0]], dtype=torch.float32),
            },
        }
    )
    regularization = {
        "consistency_cascade_weight": 1.0,
        "consistency_assay_affinity_weight": 1.0,
        "consistency_assay_presentation_weight": 1.0,
        "consistency_no_b2m_weight": 1.0,
        "consistency_prob_margin": 0.0,
    }

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=regularization,
    )

    assert torch.isfinite(total)
    assert float(total.item()) > 0.0
    assert float(losses["consistency_cascade"].item()) > 0.0
    assert float(losses["consistency_affinity_heads"].item()) > 0.0
    assert float(losses["consistency_assay_presentation"].item()) > 0.0
    assert float(losses["consistency_chain_assembly"].item()) > 0.0


def test_compute_loss_applies_tcell_upstream_prior():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class="I",
        tcell_label=1.0,
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])
    model = _DummyOutputsModel(
        {
            "processing_logit": torch.tensor([[-1.0]], dtype=torch.float32),
            "binding_logit": torch.tensor([-3.0], dtype=torch.float32),
            "presentation_logit": torch.tensor([[-2.0]], dtype=torch.float32),
            "elution_logit": torch.zeros((1, 1), dtype=torch.float32),
            "ms_logit": torch.zeros((1, 1), dtype=torch.float32),
            "immunogenicity_logit": torch.tensor([0.0], dtype=torch.float32),
            "tcell_logit": torch.tensor([[3.0]], dtype=torch.float32),
            "tcell_context_logits": {
                "assay_method": torch.zeros((1, len(TCELL_ASSAY_METHODS)), dtype=torch.float32),
                "assay_readout": torch.zeros((1, len(TCELL_ASSAY_READOUTS)), dtype=torch.float32),
                "apc_type": torch.zeros((1, len(TCELL_APC_TYPES)), dtype=torch.float32),
                "culture_context": torch.zeros((1, len(TCELL_CULTURE_CONTEXTS)), dtype=torch.float32),
                "stim_context": torch.zeros((1, len(TCELL_STIM_CONTEXTS)), dtype=torch.float32),
            },
            "assays": {
                "KD_nM": torch.zeros((1, 1), dtype=torch.float32),
                "IC50_nM": torch.zeros((1, 1), dtype=torch.float32),
                "EC50_nM": torch.zeros((1, 1), dtype=torch.float32),
            },
        }
    )
    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization={"consistency_tcell_upstream_weight": 1.0, "consistency_prob_margin": 0.0},
    )
    assert torch.isfinite(total)
    assert "consistency_tcell_upstream" in losses
    assert float(losses["consistency_tcell_upstream"].item()) > 0.0


def test_compute_loss_applies_tcell_context_prior():
    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            tcell_label=1.0,
            tcell_in_vitro_process="Primary induction in vitro",
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            tcell_label=1.0,
            tcell_effector_culture="Direct ex vivo",
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)
    model = _DummyOutputsModel(
        {
            "processing_logit": torch.zeros((2, 1), dtype=torch.float32),
            "binding_logit": torch.zeros(2, dtype=torch.float32),
            "presentation_logit": torch.zeros((2, 1), dtype=torch.float32),
            "elution_logit": torch.zeros((2, 1), dtype=torch.float32),
            "ms_logit": torch.zeros((2, 1), dtype=torch.float32),
            "immunogenicity_logit": torch.zeros(2, dtype=torch.float32),
            "tcell_logit": torch.tensor([[-1.0], [2.0]], dtype=torch.float32),
            "tcell_context_logits": {
                "assay_method": torch.zeros((2, len(TCELL_ASSAY_METHODS)), dtype=torch.float32),
                "assay_readout": torch.zeros((2, len(TCELL_ASSAY_READOUTS)), dtype=torch.float32),
                "apc_type": torch.zeros((2, len(TCELL_APC_TYPES)), dtype=torch.float32),
                "culture_context": torch.zeros((2, len(TCELL_CULTURE_CONTEXTS)), dtype=torch.float32),
                "stim_context": torch.zeros((2, len(TCELL_STIM_CONTEXTS)), dtype=torch.float32),
            },
            "assays": {
                "KD_nM": torch.zeros((2, 1), dtype=torch.float32),
                "IC50_nM": torch.zeros((2, 1), dtype=torch.float32),
                "EC50_nM": torch.zeros((2, 1), dtype=torch.float32),
            },
        }
    )
    regularization = {
        "consistency_tcell_context_weight": 1.0,
        "tcell_in_vitro_margin": 0.2,
        "tcell_ex_vivo_margin": 0.0,
    }

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=regularization,
    )

    assert torch.isfinite(total)
    assert "consistency_tcell_context" in losses
    assert float(losses["consistency_tcell_context"].item()) > 0.0


def test_compute_loss_uses_mil_noisy_or_for_elution_family_tasks():
    class _MilModel(nn.Module):
        def forward(self, **kwargs):
            n = kwargs["pep_tok"].shape[0]
            if n == 3:
                # MIL instance call: bag0 has two instances, bag1 has one.
                return {
                    "elution_logit": torch.tensor([[3.0], [-3.0], [0.5]], dtype=torch.float32),
                    "presentation_logit": torch.tensor([[2.5], [-2.0], [0.25]], dtype=torch.float32),
                    "ms_logit": torch.tensor([[2.0], [-1.5], [0.0]], dtype=torch.float32),
                }
            # Main per-sample call (not used for elution/presentation/ms when MIL is present).
            return {
                "elution_logit": torch.zeros((n, 1), dtype=torch.float32),
                "presentation_logit": torch.zeros((n, 1), dtype=torch.float32),
                "ms_logit": torch.zeros((n, 1), dtype=torch.float32),
            }

    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTA",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=1.0,
            species="human",
            mil_mhc_a_list=["MAVMAPRTLLLLLSGALALTQTA", "CAVMAPRTLLLLLSGALALTQTA"],
            mil_mhc_b_list=["MSRSVALAVLALLSLSGLEA", "MSRSVALAVLALLSLSGLEA"],
            mil_mhc_class_list=["I", "I"],
            mil_species_list=["human", "human"],
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="DAVMAPRTLLLLLSGALALTQTA",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=0.0,
            species="human",
            mil_mhc_a_list=["DAVMAPRTLLLLLSGALALTQTA"],
            mil_mhc_b_list=["MSRSVALAVLALLSLSGLEA"],
            mil_mhc_class_list=["I"],
            mil_species_list=["human"],
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)

    total, losses, metrics = compute_loss(
        model=_MilModel(),
        batch=batch,
        device="cpu",
        regularization=None,
    )

    assert torch.isfinite(total)
    assert "elution" in losses
    assert "presentation" in losses
    assert "ms" in losses
    assert float(losses["elution"].item()) > 0.0
    assert float(losses["presentation"].item()) > 0.0
    assert float(losses["ms"].item()) > 0.0
    assert "out_mil_elution_prob_mean" in metrics
    assert "out_mil_presentation_prob_mean" in metrics
    assert "out_mil_ms_prob_mean" in metrics


def test_compute_loss_uses_tcell_pathway_mil_with_context():
    class _TCellMilModel(nn.Module):
        def forward(self, **kwargs):
            n = kwargs["pep_tok"].shape[0]
            context = kwargs.get("tcell_context")
            if n == 2:
                assert context is not None
                assert context["assay_method_idx"].shape[0] == 2
                return {
                    "tcell_logit": torch.tensor([[2.0], [-1.0]], dtype=torch.float32),
                    "immunogenicity_logit": torch.tensor([[1.5], [-0.5]], dtype=torch.float32),
                }
            return {
                "tcell_logit": torch.zeros((n, 1), dtype=torch.float32),
                "immunogenicity_logit": torch.zeros((n, 1), dtype=torch.float32),
            }

    sample = PrestoSample(
        peptide="ACDEFGHIKLMN",
        mhc_a="MAVMAPRTLLLLLSGALALTQTA",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class=None,
        tcell_label=1.0,
        tcell_assay_method="ELISPOT",
        tcell_assay_readout="IFNg release",
        tcell_in_vitro_responder="PBMC",
        use_tcell_pathway_mil=True,
        tcell_mil_mhc_a_list=[
            "MAVMAPRTLLLLLSGALALTQTA",
            "DAVMAPRTLLLLLSGALALTQTA",
        ],
        tcell_mil_mhc_b_list=[
            "MSRSVALAVLALLSLSGLEA",
            "",
        ],
        tcell_mil_mhc_class_list=["I", "II"],
        tcell_mil_species_list=["human", "human"],
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])

    total, losses, metrics = compute_loss(
        model=_TCellMilModel(),
        batch=batch,
        device="cpu",
        regularization=None,
    )

    assert torch.isfinite(total)
    assert "tcell_mil" in losses
    assert "immunogenicity_mil" in losses
    assert float(losses["tcell_mil"].item()) > 0.0
    assert float(losses["immunogenicity_mil"].item()) > 0.0
    assert "out_tcell_mil_tcell_mil_prob_mean" in metrics


def test_compute_loss_adds_mil_contrastive_and_sparsity_penalties():
    class _ContrastiveMilModel(nn.Module):
        def forward(self, **kwargs):
            mhc_signal = kwargs["mhc_a_tok"][:, 0].float().unsqueeze(-1) / 10.0
            return {
                "elution_logit": mhc_signal,
                "presentation_logit": mhc_signal,
                "ms_logit": mhc_signal,
            }

    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MMMMMMMMMMMMMMMMMMMMMMMM",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=1.0,
            species="human",
            mil_mhc_a_list=[
                "MMMMMMMMMMMMMMMMMMMMMMMM",
                "NNNNNNNNNNNNNNNNNNNNNNNN",
            ],
            mil_mhc_b_list=[
                "MSRSVALAVLALLSLSGLEA",
                "MSRSVALAVLALLSLSGLEA",
            ],
            mil_mhc_class_list=["I", "I"],
            mil_species_list=["human", "human"],
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="YYYYYYYYYYYYYYYYYYYYYYYY",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=0.0,
            species="human",
            mil_mhc_a_list=[
                "YYYYYYYYYYYYYYYYYYYYYYYY",
                "WWWWWWWWWWWWWWWWWWWWWWWW",
            ],
            mil_mhc_b_list=[
                "MSRSVALAVLALLSLSGLEA",
                "MSRSVALAVLALLSLSGLEA",
            ],
            mil_mhc_class_list=["I", "I"],
            mil_species_list=["human", "human"],
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)

    total, losses, metrics = compute_loss(
        model=_ContrastiveMilModel(),
        batch=batch,
        device="cpu",
        regularization={
            "mil_contrastive_weight": 1.0,
            "mil_contrastive_margin": 0.5,
            "mil_contrastive_max_pairs": 4,
            "mil_bag_sparsity_weight": 1.0,
            "mil_bag_sparsity_target_sum": 1.5,
        },
    )

    assert torch.isfinite(total)
    assert "presentation_mil_contrastive" in losses
    assert "elution_mil_sparsity" in losses
    assert "presentation_mil_sparsity" in losses
    assert "ms_mil_sparsity" in losses
    assert float(losses["presentation_mil_contrastive"].item()) > 0.0
    assert float(losses["presentation_mil_sparsity"].item()) > 0.0
    assert metrics["out_mil_contrastive_pairs"] == pytest.approx(1.0)


def test_mil_instance_cap_preserves_at_least_one_instance_per_bag():
    class _TrackingMilModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.call_ns = []

        def forward(self, **kwargs):
            n = kwargs["pep_tok"].shape[0]
            self.call_ns.append(n)
            return {
                "elution_logit": torch.zeros((n, 1), dtype=torch.float32),
                "presentation_logit": torch.zeros((n, 1), dtype=torch.float32),
                "ms_logit": torch.zeros((n, 1), dtype=torch.float32),
            }

    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTA",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=1.0,
            species="human",
            mil_mhc_a_list=[
                "MAVMAPRTLLLLLSGALALTQTA",
                "CAVMAPRTLLLLLSGALALTQTA",
            ],
            mil_mhc_b_list=["MSRSVALAVLALLSLSGLEA", "MSRSVALAVLALLSLSGLEA"],
            mil_mhc_class_list=["I", "I"],
            mil_species_list=["human", "human"],
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="DAVMAPRTLLLLLSGALALTQTA",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            elution_label=0.0,
            species="human",
            mil_mhc_a_list=[
                "DAVMAPRTLLLLLSGALALTQTA",
                "EAVMAPRTLLLLLSGALALTQTA",
            ],
            mil_mhc_b_list=["MSRSVALAVLALLSLSGLEA", "MSRSVALAVLALLSLSGLEA"],
            mil_mhc_class_list=["I", "I"],
            mil_species_list=["human", "human"],
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)
    model = _TrackingMilModel()

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=None,
        max_mil_instances=1,
    )

    assert torch.isfinite(total)
    assert "elution" in losses
    assert len(model.call_ns) >= 2
    assert model.call_ns[1] >= 2


def test_compute_loss_adds_binding_orthogonality_regularization():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class="I",
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])

    aligned_model = _DummyOutputsModel(
        {
            "latent_vecs": {
                "binding_affinity": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                "binding_stability": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            }
        }
    )
    orth_model = _DummyOutputsModel(
        {
            "latent_vecs": {
                "binding_affinity": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                "binding_stability": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            }
        }
    )

    total_aligned, losses_aligned, _ = compute_loss(
        model=aligned_model,
        batch=batch,
        device="cpu",
        regularization={"binding_orthogonality_weight": 1.0},
    )
    total_orth, losses_orth, _ = compute_loss(
        model=orth_model,
        batch=batch,
        device="cpu",
        regularization={"binding_orthogonality_weight": 1.0},
    )

    assert torch.isfinite(total_aligned)
    assert torch.isfinite(total_orth)
    assert "consistency_binding_orthogonality" in losses_aligned
    assert "consistency_binding_orthogonality" in losses_orth
    assert float(losses_aligned["consistency_binding_orthogonality"].item()) > 0.9
    assert losses_orth["consistency_binding_orthogonality"].item() == pytest.approx(0.0, abs=1e-6)


def test_compute_loss_includes_core_start_ce_when_labels_present():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class="II",
        core_start=1,
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])
    logits = torch.full((1, 16), -4.0, dtype=torch.float32)
    logits[0, 1] = 4.0
    model = _DummyOutputsModel({"core_start_logit": logits})

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=None,
    )

    assert torch.isfinite(total)
    assert "core_start" in losses
    assert float(losses["core_start"].item()) >= 0.0


def test_compute_loss_adds_binding_mhc_attention_sparsity_penalty_when_out_of_range():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class="I",
        bind_value=10000.0,
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])
    model = _DummyOutputsModel(
        {
            "binding_logit": torch.tensor([-3.0], dtype=torch.float32),
            "binding_mhc_attention_effective_residues": torch.tensor([12.0], dtype=torch.float32),
            "binding_mhc_attention_valid_mask": torch.tensor([1.0], dtype=torch.float32),
        }
    )

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization={
            "mhc_attention_sparsity_weight": 0.5,
            "mhc_attention_sparsity_min_residues": 30.0,
            "mhc_attention_sparsity_max_residues": 60.0,
        },
    )

    assert torch.isfinite(total)
    assert "consistency_binding_mhc_attention_sparsity" in losses
    assert float(losses["consistency_binding_mhc_attention_sparsity"].item()) > 0.0


def test_compute_loss_skips_binding_mhc_attention_sparsity_penalty_in_target_band():
    sample = PrestoSample(
        peptide="SIINFEKL",
        mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
        mhc_b="MSRSVALAVLALLSLSGLEA",
        mhc_class="I",
        bind_value=10000.0,
    )
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)([sample])
    model = _DummyOutputsModel(
        {
            "binding_logit": torch.tensor([-3.0], dtype=torch.float32),
            "binding_mhc_attention_effective_residues": torch.tensor([45.0], dtype=torch.float32),
            "binding_mhc_attention_valid_mask": torch.tensor([1.0], dtype=torch.float32),
        }
    )

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization={
            "mhc_attention_sparsity_weight": 0.5,
            "mhc_attention_sparsity_min_residues": 30.0,
            "mhc_attention_sparsity_max_residues": 60.0,
        },
    )

    assert torch.isfinite(total)
    assert "consistency_binding_mhc_attention_sparsity" in losses
    assert losses["consistency_binding_mhc_attention_sparsity"].item() == pytest.approx(0.0)


def test_compute_loss_support_weighted_aggregation_reduces_rare_task_influence():
    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            bind_value=10_000.0,
            tcell_label=1.0,
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            bind_value=10_000.0,
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)
    model = _DummyOutputsModel(
        {
            "assays": {"KD_nM": torch.tensor([[2.0], [2.0]], dtype=torch.float32)},
            "tcell_logit": torch.tensor([[-10.0], [0.0]], dtype=torch.float32),
        }
    )

    total_task_mean, losses_task_mean, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=None,
        supervised_loss_aggregation="task_mean",
    )
    total_sample_weighted, losses_sample_weighted, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=None,
        supervised_loss_aggregation="sample_weighted",
    )

    assert torch.isfinite(total_task_mean)
    assert torch.isfinite(total_sample_weighted)
    assert "binding" in losses_task_mean
    assert "tcell" in losses_task_mean
    assert losses_task_mean.keys() == losses_sample_weighted.keys()

    binding_loss = float(losses_task_mean["binding"].item())
    tcell_loss = float(losses_task_mean["tcell"].item())
    expected_task_mean = 0.5 * (binding_loss + tcell_loss)
    expected_sample_weighted = (2.0 * binding_loss + tcell_loss) / 3.0

    assert float(total_task_mean.item()) == pytest.approx(expected_task_mean, rel=1e-6)
    assert float(total_sample_weighted.item()) == pytest.approx(
        expected_sample_weighted, rel=1e-6
    )
    assert float(total_sample_weighted.item()) < float(total_task_mean.item())


def test_compute_loss_applies_auxiliary_base_weights_before_support_aggregation():
    samples = [
        PrestoSample(
            peptide="SIINFEKL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            bind_value=10_000.0,
            primary_allele="HLA-A*02:01",
        ),
        PrestoSample(
            peptide="GILGFVFTL",
            mhc_a="MAVMAPRTLLLLLSGALALTQTWAG",
            mhc_b="MSRSVALAVLALLSLSGLEA",
            mhc_class="I",
            primary_allele="HLA-A*02:01",
        ),
    ]
    batch = PrestoCollator(max_pep_len=16, max_mhc_len=64)(samples)
    model = _DummyOutputsModel(
        {
            "assays": {"KD_nM": torch.tensor([[2.0], [2.0]], dtype=torch.float32)},
            "mhc_class_logits": torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=torch.float32),
        }
    )

    total, losses, _ = compute_loss(
        model=model,
        batch=batch,
        device="cpu",
        regularization=None,
        supervised_loss_aggregation="sample_weighted",
    )

    assert torch.isfinite(total)
    assert set(losses) == {"binding", "mhc_class"}

    binding_loss = float(losses["binding"].item())
    mhc_class_loss = float(losses["mhc_class"].item())
    expected = (binding_loss * 1.0 + mhc_class_loss * 0.1 * 2.0) / (1.0 + 0.1 * 2.0)
    assert float(total.item()) == pytest.approx(expected, rel=1e-6)


def test_build_warmup_cosine_scheduler_warms_then_decays():
    param = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=1e-3,
        total_steps=40,
    )

    assert scheduler is not None
    lrs = []
    for _ in range(40):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert lrs[0] < lrs[1]
    assert max(lrs) <= 1e-3 * (1.0 + 1e-6)
    assert lrs[-1] < lrs[1]
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-3)


def test_train_epoch_steps_scheduler(monkeypatch):
    class _ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

    model = _ToyModel()
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        base_lr=base_lr,
        total_steps=2,
    )
    batches = [
        types.SimpleNamespace(pep_tok=torch.ones((1, 1), dtype=torch.long)),
        types.SimpleNamespace(pep_tok=torch.ones((1, 1), dtype=torch.long)),
    ]

    def _fake_compute_loss(model, batch, device, uncertainty_weighting=None, **kwargs):
        loss = model.weight.square()
        return loss, {"toy": loss}, {}

    monkeypatch.setattr("presto.scripts.train_synthetic.compute_loss", _fake_compute_loss)
    initial_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, metrics = train_epoch(
        model,
        batches,
        optimizer,
        device="cpu",
        scheduler=scheduler,
        show_progress=False,
    )

    assert train_loss > 0.0
    assert metrics["train_batches"] == 2.0
    assert float(optimizer.param_groups[0]["lr"]) > initial_lr
    assert float(optimizer.param_groups[0]["lr"]) == pytest.approx(base_lr * 0.1, rel=1e-3)
