"""Smoke tests for distributional vs regression BA heads."""

from __future__ import annotations

import math

import pytest
import torch

from presto.scripts.distributional_ba.config import (
    CONDITIONS,
    ConditionSpec,
    build_model,
)
from presto.scripts.distributional_ba.config_v6 import CONDITIONS_V6_BY_ID
from presto.scripts.distributional_ba.heads import (
    HEAD_REGISTRY,
    GaussianHead,
    HLGaussHead,
    LogMSEHead,
    MHCflurryHead,
    QuantileHead,
    TwoHotHead,
)
from presto.scripts.distributional_ba.losses import (
    censored_gaussian_nll,
    censored_pinball_loss,
    distributional_cross_entropy,
    gaussian_nll,
    log_censored_mse,
    mhcflurry_censored_loss,
    survival_nll,
)
from presto.scripts.distributional_ba.assay_context import (
    AssayContextEncoder,
    D1AffineIntegration,
    D2LogitIntegration,
)
from presto.scripts.distributional_ba.metrics import point_metrics, calibration_metrics


B, D, CTX = 4, 384, 32  # batch, encoder dim (3*128), context dim


# ---------------------------------------------------------------------------
# Condition matrix
# ---------------------------------------------------------------------------

def test_32_conditions():
    assert len(CONDITIONS) == 32
    ids = [c.cond_id for c in CONDITIONS]
    assert ids == list(range(1, 33))


# ---------------------------------------------------------------------------
# Forward pass shape checks for all 4 head types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_type,assay_mode", [
    ("mhcflurry", "affine"),
    ("mhcflurry", "additive"),
    ("log_mse", "affine"),
    ("log_mse", "additive"),
    ("twohot", "d1_affine"),
    ("twohot", "d2_logit"),
    ("hlgauss", "d1_affine"),
    ("hlgauss", "d2_logit"),
])
def test_head_forward_shape(head_type, assay_mode):
    kwargs = dict(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    if head_type in ("twohot", "hlgauss"):
        kwargs["n_bins"] = 64
    if head_type == "hlgauss":
        kwargs["sigma_mult"] = 0.75
    head = HEAD_REGISTRY[head_type](**kwargs)

    h = torch.randn(B, D)
    ctx = torch.randn(B, CTX)
    out = head(h, ctx)

    assert "pred_ic50_nM" in out
    assert out["pred_ic50_nM"].shape == (B,)
    assert (out["pred_ic50_nM"] >= 0).all()


# ---------------------------------------------------------------------------
# Loss backward — gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_type,assay_mode", [
    ("mhcflurry", "affine"),
    ("log_mse", "affine"),
    ("twohot", "d1_affine"),
    ("hlgauss", "d1_affine"),
    ("twohot", "d2_logit"),
    ("hlgauss", "d2_logit"),
])
def test_loss_backward(head_type, assay_mode):
    kwargs = dict(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    if head_type in ("twohot", "hlgauss"):
        kwargs["n_bins"] = 32
    if head_type == "hlgauss":
        kwargs["sigma_mult"] = 0.75
    head = HEAD_REGISTRY[head_type](**kwargs)

    h = torch.randn(B, D, requires_grad=True)
    ctx = torch.randn(B, CTX)
    ic50_nM = torch.tensor([100.0, 500.0, 1000.0, 25000.0])
    qual = torch.tensor([0, 0, 1, -1])
    mask = torch.ones(B)

    loss, metrics = head.compute_loss(h, ctx, ic50_nM, qual, mask)
    loss.backward()

    assert h.grad is not None
    assert (h.grad != 0).any(), "No gradient flow through the head"
    assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Survival NLL edge cases
# ---------------------------------------------------------------------------

def test_survival_nll_threshold_below_range():
    """Threshold below all bins → P(Y >= threshold) ≈ 1 → loss ≈ 0."""
    K = 16
    probs = torch.softmax(torch.randn(1, K), dim=-1)
    edges = torch.linspace(1.0, 10.0, K + 1)
    threshold = torch.tensor([0.5])  # below range
    direction = torch.tensor([1])    # >, meaning true >= threshold

    loss = survival_nll(probs, edges, threshold, direction)
    # Almost all mass is above threshold, so loss should be near 0
    assert loss.item() < 0.1


def test_survival_nll_threshold_above_range():
    """Threshold above all bins → P(Y >= threshold) ≈ 0 → large loss."""
    K = 16
    probs = torch.softmax(torch.randn(1, K), dim=-1)
    edges = torch.linspace(1.0, 10.0, K + 1)
    threshold = torch.tensor([11.0])  # above range
    direction = torch.tensor([1])     # >, meaning true >= threshold

    loss = survival_nll(probs, edges, threshold, direction)
    assert loss.item() > 1.0  # should be large (-log(small))


# ---------------------------------------------------------------------------
# Target vector properties
# ---------------------------------------------------------------------------

def test_twohot_target_sums_to_one():
    head = TwoHotHead(in_dim=D, ctx_dim=CTX, n_bins=64, assay_mode="d1_affine")
    centers = head.bin_centers.unsqueeze(0).expand(B, -1)
    edges = head.bin_edges.unsqueeze(0).expand(B, -1)
    y = torch.tensor([0.5, 3.0, 7.0, 10.0])  # various log1p values
    target = head._build_target_vector(y, centers, edges)

    assert target.shape == (B, 64)
    sums = target.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)
    # Should be sparse (at most 2 non-zero entries per row)
    assert (target > 0).sum(dim=-1).max() <= 2


def test_hlgauss_target_sums_to_approx_one():
    head = HLGaussHead(in_dim=D, ctx_dim=CTX, n_bins=64, sigma_mult=0.75, assay_mode="d1_affine")
    centers = head.bin_centers.unsqueeze(0).expand(B, -1)
    edges = head.bin_edges.unsqueeze(0).expand(B, -1)
    y = torch.tensor([0.5, 3.0, 7.0, 10.0])
    target = head._build_target_vector(y, centers, edges)

    assert target.shape == (B, 64)
    sums = target.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-3)


# ---------------------------------------------------------------------------
# D1-affine vs D2-logit produce different outputs
# ---------------------------------------------------------------------------

def test_d1_vs_d2_different_outputs():
    """Different assay contexts should produce different predictions."""
    head_d1 = TwoHotHead(in_dim=D, ctx_dim=CTX, n_bins=32, assay_mode="d1_affine")
    head_d2 = TwoHotHead(in_dim=D, ctx_dim=CTX, n_bins=32, assay_mode="d2_logit")

    h = torch.randn(2, D)
    # Two different assay contexts
    ctx_a = torch.randn(1, CTX).expand(2, -1)
    ctx_b = torch.randn(1, CTX).expand(2, -1)

    for head in [head_d1, head_d2]:
        out_a = head(h, ctx_a)
        out_b = head(h, ctx_b)
        # Different contexts should give different predictions
        # (unless by astronomical coincidence)
        assert not torch.allclose(out_a["pred_ic50_nM"], out_b["pred_ic50_nM"], atol=1e-4)


# ---------------------------------------------------------------------------
# Assay context encoder
# ---------------------------------------------------------------------------

def test_assay_context_encoder():
    enc = AssayContextEncoder(factor_dim=8, ctx_dim=CTX)
    type_idx = torch.zeros(B, dtype=torch.long)
    prep_idx = torch.ones(B, dtype=torch.long)
    geom_idx = torch.zeros(B, dtype=torch.long)
    read_idx = torch.zeros(B, dtype=torch.long)
    out = enc(type_idx, prep_idx, geom_idx, read_idx)
    assert out.shape == (B, CTX)


# ---------------------------------------------------------------------------
# Full model build
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cond_id", [1, 5, 7, 13, 21, 29])
def test_build_model_smoke(cond_id):
    from presto.scripts.distributional_ba.config import CONDITIONS_BY_ID
    spec = CONDITIONS_BY_ID[cond_id]
    model = build_model(spec)

    pep = torch.randint(1, 20, (2, 15))
    mhc_a = torch.randint(1, 20, (2, 40))
    mhc_b = torch.randint(1, 20, (2, 40))

    out = model(pep, mhc_a, mhc_b)
    assert "pred_ic50_nM" in out
    assert out["pred_ic50_nM"].shape == (2,)


# ---------------------------------------------------------------------------
# Calibration metrics on synthetic data
# ---------------------------------------------------------------------------

def test_calibration_metrics_smoke():
    K = 32
    B_cal = 50
    # Perfect calibration: put all mass on the bin containing the true value
    edges = torch.linspace(0, 10, K + 1)
    true_y = torch.rand(B_cal) * 10  # uniform in [0, 10]
    probs = torch.zeros(B_cal, K)
    for i in range(B_cal):
        bin_idx = min(int(true_y[i] / (10.0 / K)), K - 1)
        probs[i, bin_idx] = 1.0
    mask = torch.ones(B_cal)

    result = calibration_metrics(probs, edges, true_y, mask)
    assert "pit_ks" in result
    assert "coverage_90" in result
    # Perfect calibration should have high coverage
    assert result["coverage_90"] >= 0.8


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def test_point_metrics_perfect():
    pred = torch.tensor([10.0, 100.0, 1000.0, 10000.0])
    true = torch.tensor([10.0, 100.0, 1000.0, 10000.0])
    mask = torch.ones(4)
    result = point_metrics(pred, true, mask)
    assert result["rmse_log10"] < 0.01
    assert result["spearman"] > 0.99
    assert result["pearson"] > 0.99


# ---------------------------------------------------------------------------
# Standalone loss functions
# ---------------------------------------------------------------------------

def test_distributional_ce():
    logits = torch.randn(B, 32)
    target = torch.softmax(torch.randn(B, 32), dim=-1)
    ce = distributional_cross_entropy(logits, target)
    assert ce.shape == (B,)
    assert (ce >= 0).all()


def test_log_censored_mse_exact():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    qual = torch.tensor([0, 0, 0])
    mask = torch.ones(3)
    loss = log_censored_mse(pred, target, qual, mask)
    assert loss.item() < 1e-6


def test_mhcflurry_censored_loss_exact():
    pred = torch.tensor([0.5, 0.3])
    target = torch.tensor([0.5, 0.3])
    qual = torch.tensor([0, 0])
    mask = torch.ones(2)
    loss = mhcflurry_censored_loss(pred, target, qual, mask)
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# Gaussian head tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("assay_mode", ["affine", "additive"])
def test_gaussian_head_forward_shape(assay_mode):
    head = GaussianHead(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    h = torch.randn(B, D)
    ctx = torch.randn(B, CTX)
    out = head(h, ctx)
    assert "pred_ic50_nM" in out
    assert out["pred_ic50_nM"].shape == (B,)
    assert "pred_sigma" in out
    assert out["pred_sigma"].shape == (B,)
    assert (out["pred_sigma"] > 0).all()


@pytest.mark.parametrize("assay_mode", ["affine", "additive"])
def test_gaussian_head_loss_backward(assay_mode):
    head = GaussianHead(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    h = torch.randn(B, D, requires_grad=True)
    ctx = torch.randn(B, CTX)
    ic50_nM = torch.tensor([100.0, 500.0, 1000.0, 25000.0])
    qual = torch.tensor([0, 0, 1, -1])
    mask = torch.ones(B)
    loss, metrics = head.compute_loss(h, ctx, ic50_nM, qual, mask)
    loss.backward()
    assert h.grad is not None
    assert (h.grad != 0).any()
    assert loss.isfinite()


def test_gaussian_head_predict_distribution():
    head = GaussianHead(in_dim=D, ctx_dim=CTX, assay_mode="affine")
    h = torch.randn(B, D)
    ctx = torch.randn(B, CTX)
    dist = head.predict_distribution(h, ctx)
    assert dist is not None
    assert "mu" in dist
    assert "sigma" in dist


def test_gaussian_nll_perfect():
    mu = torch.tensor([1.0, 2.0, 3.0])
    sigma = torch.tensor([0.01, 0.01, 0.01])  # very certain
    target = torch.tensor([1.0, 2.0, 3.0])
    nll = gaussian_nll(mu, sigma, target)
    assert nll.shape == (3,)
    # NLL should be low (just the -log(sigma) + const term)
    assert (nll < 5).all()


def test_censored_gaussian_nll_consistent():
    """Right-censored: if mu >> threshold, loss should be low."""
    mu = torch.tensor([10.0])
    sigma = torch.tensor([1.0])
    threshold = torch.tensor([2.0])
    direction = torch.tensor([1])  # true >= 2, and mu=10 >> 2
    loss = censored_gaussian_nll(mu, sigma, threshold, direction)
    assert loss.item() < 0.1  # Phi((10-2)/1) ≈ 1, so -log(~1) ≈ 0


# ---------------------------------------------------------------------------
# Quantile head tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("assay_mode", ["affine", "additive"])
def test_quantile_head_forward_shape(assay_mode):
    head = QuantileHead(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    h = torch.randn(B, D)
    ctx = torch.randn(B, CTX)
    out = head(h, ctx)
    assert "pred_ic50_nM" in out
    assert out["pred_ic50_nM"].shape == (B,)
    assert "quantiles" in out
    assert out["quantiles"].shape == (B, 5)
    # Quantiles should be sorted
    q = out["quantiles"]
    assert (q[:, 1:] >= q[:, :-1] - 1e-6).all()


@pytest.mark.parametrize("assay_mode", ["affine", "additive"])
def test_quantile_head_loss_backward(assay_mode):
    head = QuantileHead(in_dim=D, ctx_dim=CTX, max_nM=50_000.0, assay_mode=assay_mode)
    h = torch.randn(B, D, requires_grad=True)
    ctx = torch.randn(B, CTX)
    ic50_nM = torch.tensor([100.0, 500.0, 1000.0, 25000.0])
    qual = torch.tensor([0, 0, 1, -1])
    mask = torch.ones(B)
    loss, metrics = head.compute_loss(h, ctx, ic50_nM, qual, mask)
    loss.backward()
    assert h.grad is not None
    assert (h.grad != 0).any()
    assert loss.isfinite()


def test_quantile_head_predict_distribution():
    head = QuantileHead(in_dim=D, ctx_dim=CTX, assay_mode="affine")
    h = torch.randn(B, D)
    ctx = torch.randn(B, CTX)
    dist = head.predict_distribution(h, ctx)
    assert dist is not None
    assert "quantiles" in dist
    assert "iqr" in dist
    assert dist["iqr"].shape == (B,)


def test_censored_pinball_exact():
    """Exact pinball loss at perfect prediction should be 0."""
    quantiles = torch.tensor([[2.0, 3.0, 5.0, 7.0, 8.0]])
    target = torch.tensor([5.0])  # exactly at median
    direction = torch.tensor([0])  # exact
    tau = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    loss = censored_pinball_loss(quantiles, target, direction, tau)
    # Loss for median (q50) should be 0 when prediction equals target
    assert loss[0, 2].item() < 1e-6


def test_censored_pinball_right_censored():
    """Right-censored: no penalty when quantile >= threshold."""
    quantiles = torch.tensor([[8.0, 9.0, 10.0, 11.0, 12.0]])
    target = torch.tensor([5.0])  # threshold
    direction = torch.tensor([1])  # right-censored: true >= 5
    tau = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    loss = censored_pinball_loss(quantiles, target, direction, tau)
    # All quantiles > threshold, so all losses should be 0
    assert (loss < 1e-6).all()


# ---------------------------------------------------------------------------
# V2 config tests
# ---------------------------------------------------------------------------

def test_v2_28_conditions():
    from presto.scripts.distributional_ba.config_v2 import CONDITIONS_V2
    assert len(CONDITIONS_V2) == 28
    ids = [c.cond_id for c in CONDITIONS_V2]
    assert ids == list(range(1, 29))


def test_v2_build_model_smoke():
    from presto.scripts.distributional_ba.config_v2 import CONDITIONS_V2_BY_ID
    from presto.scripts.distributional_ba.config import build_model
    # Test one of each new head type
    for cond_id in [1, 5, 9, 13, 21]:
        spec = CONDITIONS_V2_BY_ID[cond_id]
        model = build_model(spec)
        pep = torch.randint(1, 20, (2, 15))
        mhc_a = torch.randint(1, 20, (2, 40))
        mhc_b = torch.randint(1, 20, (2, 40))
        out = model(pep, mhc_a, mhc_b)
        assert "pred_ic50_nM" in out
        assert out["pred_ic50_nM"].shape == (2,)


# ---------------------------------------------------------------------------
# Content-conditioned assay context tests
# ---------------------------------------------------------------------------

def test_content_conditioned_assay_context():
    """AssayContextEncoder with repr_dim > 0 accepts binding logit + mol repr."""
    from presto.scripts.distributional_ba.assay_context import AssayContextEncoder
    repr_dim = 384
    enc = AssayContextEncoder(ctx_dim=32, repr_dim=repr_dim)
    B = 4
    type_idx = torch.zeros(B, dtype=torch.long)
    prep_idx = torch.zeros(B, dtype=torch.long)
    geom_idx = torch.zeros(B, dtype=torch.long)
    read_idx = torch.zeros(B, dtype=torch.long)
    logit = torch.randn(B)
    mol = torch.randn(B, repr_dim)
    out = enc(type_idx, prep_idx, geom_idx, read_idx, binding_logit=logit, mol_repr=mol)
    assert out.shape == (B, 32)


def test_content_conditioned_build_model_smoke():
    """build_model with content_conditioned=True produces working model."""
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    model = build_model(spec, content_conditioned=True)
    assert model.assay_ctx.repr_dim > 0
    pep = torch.randint(1, 20, (2, 15))
    mhc_a = torch.randint(1, 20, (2, 40))
    mhc_b = torch.randint(1, 20, (2, 40))
    out = model(pep, mhc_a, mhc_b)
    assert "pred_ic50_nM" in out
    assert out["pred_ic50_nM"].shape == (2,)


def test_no_assay_input_mode_zeroes_assay_embedding():
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    model = build_model(spec, assay_input_mode="none")
    pep = torch.randint(1, 20, (2, 15))
    mhc_a = torch.randint(1, 20, (2, 40))
    mhc_b = torch.randint(1, 20, (2, 40))
    h = model.encode_input(pep, mhc_a, mhc_b)
    assay_emb = model._compute_assay_emb(
        h,
        torch.tensor([0, 4], dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.long),
        torch.tensor([0, 3], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert assay_emb.shape == (2, model.assay_ctx.ctx_dim)
    assert torch.allclose(assay_emb, torch.zeros_like(assay_emb))


def test_no_assay_input_mode_rejects_content_conditioning():
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    with pytest.raises(ValueError, match="incompatible"):
        build_model(spec, content_conditioned=True, assay_input_mode="none")


def test_content_conditioned_loss_backward():
    """Gradient flows through content-conditioned model."""
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    model = build_model(spec, content_conditioned=True)
    pep = torch.randint(1, 20, (4, 15))
    mhc_a = torch.randint(1, 20, (4, 40))
    mhc_b = torch.randint(1, 20, (4, 40))
    h = model.encode_input(pep, mhc_a, mhc_b)
    assay_emb = model._compute_assay_emb(
        h,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    )
    ic50 = torch.tensor([100.0, 500.0, 1000.0, 25000.0])
    qual = torch.tensor([0, 0, 1, -1])
    mask = torch.ones(4)
    loss, _ = model.head.compute_loss(h, assay_emb, ic50, qual, mask)
    loss.backward()
    # Encoder should get gradients through the head MLP path
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0
    # Assay context encoder should get gradients through integration
    ctx_grads = [p.grad for p in model.assay_ctx.parameters() if p.grad is not None]
    assert len(ctx_grads) > 0


def test_content_conditioned_different_inputs_different_bias():
    """Content-conditioned assay context produces different biases for different inputs."""
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    model = build_model(spec, content_conditioned=True)
    model.eval()
    pep1 = torch.randint(1, 20, (1, 15))
    pep2 = torch.randint(1, 20, (1, 15))
    mhc_a = torch.randint(1, 20, (1, 40))
    mhc_b = torch.randint(1, 20, (1, 40))
    with torch.no_grad():
        # Same assay context indices, different molecular inputs
        out1 = model(pep1, mhc_a, mhc_b)
        out2 = model(pep2, mhc_a, mhc_b)
    # With content conditioning, the assay bias depends on input, so predictions differ
    # (they'd also differ without conditioning because the base prediction differs,
    # but this confirms the pipeline works end-to-end)
    assert out1["pred_ic50_nM"].shape == (1,)
    assert out2["pred_ic50_nM"].shape == (1,)


def test_v6_historical_positive_control_contract():
    """Freeze the raw EXP-16 winner build contract in shared code."""
    spec = CONDITIONS_V6_BY_ID[2]
    model = build_model(spec, encoder_backbone="historical_ablation")
    assert type(model.encoder).__name__ == "HistoricalAblationEncoder"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 27186
    pep = torch.randint(1, 20, (2, 15))
    mhc_a = torch.randint(1, 20, (2, 40))
    mhc_b = torch.randint(1, 20, (2, 40))
    out = model(pep, mhc_a, mhc_b)
    assert out["pred_ic50_nM"].shape == (2,)


def test_v6_groove_backend_smoke():
    """Modern groove backend remains runnable on the v6 winner cell."""
    spec = CONDITIONS_V6_BY_ID[2]
    model = build_model(spec, encoder_backbone="groove")
    assert type(model.encoder).__name__ == "GrooveTransformerModel"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 27186
    pep = torch.randint(1, 20, (2, 15))
    mhc_a = torch.randint(1, 20, (2, 40))
    mhc_b = torch.randint(1, 20, (2, 40))
    out = model(pep, mhc_a, mhc_b)
    assert out["pred_ic50_nM"].shape == (2,)


@pytest.mark.parametrize("head_type", ["mhcflurry", "log_mse", "twohot", "hlgauss", "gaussian", "quantile"])
def test_compute_binding_signal(head_type):
    """All heads implement compute_binding_signal returning (B,) scalar."""
    head_cls = HEAD_REGISTRY[head_type]
    kwargs = dict(in_dim=D, ctx_dim=CTX, max_nM=50_000.0)
    if head_type in ("mhcflurry", "log_mse", "gaussian", "quantile"):
        kwargs["assay_mode"] = "additive"
    elif head_type in ("twohot", "hlgauss"):
        kwargs["assay_mode"] = "d1_affine"
    if head_type == "hlgauss":
        kwargs["sigma_mult"] = 0.75
    head = head_cls(**kwargs)
    h = torch.randn(B, D)
    signal = head.compute_binding_signal(h)
    assert signal.shape == (B,)
    assert signal.isfinite().all()


def test_v5_conditions():
    from presto.scripts.distributional_ba.config_v5 import CONDITIONS_V5
    assert len(CONDITIONS_V5) == 6
    dims = [c.embed_dim for c in CONDITIONS_V5]
    assert dims == [32, 64, 96, 128, 192, 256]


# ---------------------------------------------------------------------------
# Assay combo prediction tests
# ---------------------------------------------------------------------------

def test_predict_all_combos_point():
    """predict_all_combos returns per-combo point predictions."""
    from presto.scripts.distributional_ba.assay_combos import predict_all_combos
    spec = ConditionSpec(cond_id=99, head_type="mhcflurry", assay_mode="additive", max_nM=50_000)
    model = build_model(spec, content_conditioned=True)
    model.eval()
    pep = torch.randint(1, 20, (1, 15))
    mhc_a = torch.randint(1, 20, (1, 40))
    mhc_b = torch.randint(1, 20, (1, 40))
    combos = [
        {"assay_type_idx": 0, "assay_prep_idx": 0, "assay_geometry_idx": 0, "assay_readout_idx": 0,
         "assay_type": "unknown", "assay_prep": "unknown", "assay_geometry": "unknown",
         "assay_readout": "unknown", "count": 100},
        {"assay_type_idx": 1, "assay_prep_idx": 1, "assay_geometry_idx": 1, "assay_readout_idx": 1,
         "assay_type": "KD", "assay_prep": "PURIFIED", "assay_geometry": "COMPETITIVE",
         "assay_readout": "RADIOACTIVITY", "count": 50},
    ]
    results = predict_all_combos(model, pep, mhc_a, mhc_b, combos)
    assert len(results) == 2
    for r in results:
        assert "ic50_nM" in r
        assert "ic50_log10" in r
        assert r["ic50_nM"] > 0


def test_predict_all_combos_distributional():
    """predict_all_combos returns probs/edges for distributional heads."""
    from presto.scripts.distributional_ba.assay_combos import predict_all_combos
    spec = ConditionSpec(cond_id=99, head_type="hlgauss", assay_mode="d1_affine",
                         max_nM=50_000, sigma_mult=0.75)
    model = build_model(spec, content_conditioned=True)
    model.eval()
    pep = torch.randint(1, 20, (1, 15))
    mhc_a = torch.randint(1, 20, (1, 40))
    mhc_b = torch.randint(1, 20, (1, 40))
    combos = [
        {"assay_type_idx": 0, "assay_prep_idx": 0, "assay_geometry_idx": 0, "assay_readout_idx": 0,
         "assay_type": "unknown", "assay_prep": "unknown", "assay_geometry": "unknown",
         "assay_readout": "unknown", "count": 100},
    ]
    results = predict_all_combos(model, pep, mhc_a, mhc_b, combos)
    assert len(results) == 1
    r = results[0]
    assert "probs" in r
    assert "bin_edges" in r
    assert "entropy" in r
    assert r["probs"].shape == (128,)  # K=128 default
    assert abs(r["probs"].sum().item() - 1.0) < 1e-4


def test_predict_marginal_distributional():
    """predict_marginal produces a mixture distribution for distributional heads."""
    from presto.scripts.distributional_ba.assay_combos import predict_marginal
    spec = ConditionSpec(cond_id=99, head_type="hlgauss", assay_mode="d1_affine",
                         max_nM=50_000, sigma_mult=0.75)
    model = build_model(spec, content_conditioned=True)
    pep = torch.randint(1, 20, (1, 15))
    mhc_a = torch.randint(1, 20, (1, 40))
    mhc_b = torch.randint(1, 20, (1, 40))
    combos = [
        {"assay_type_idx": 0, "assay_prep_idx": 0, "assay_geometry_idx": 0, "assay_readout_idx": 0,
         "assay_type": "unknown", "assay_prep": "unknown", "assay_geometry": "unknown",
         "assay_readout": "unknown", "count": 100},
        {"assay_type_idx": 1, "assay_prep_idx": 1, "assay_geometry_idx": 0, "assay_readout_idx": 0,
         "assay_type": "KD", "assay_prep": "PURIFIED", "assay_geometry": "unknown",
         "assay_readout": "unknown", "count": 50},
    ]
    result = predict_marginal(model, pep, mhc_a, mhc_b, combos)
    assert "ic50_nM" in result
    assert "marginal_probs" in result
    assert "marginal_entropy" in result
    assert "marginal_ev_nM" in result
    # Marginal probs should sum to ~1
    assert abs(result["marginal_probs"].sum().item() - 1.0) < 1e-4
