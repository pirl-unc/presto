"""Tests for synthetic training loss utilities."""

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
from presto.scripts.train_synthetic import compute_loss


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
