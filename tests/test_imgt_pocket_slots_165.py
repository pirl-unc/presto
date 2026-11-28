"""Tests for pirl-unc/presto#165: exact 3D-verified 46-slot IMGT pocket contacts from
mhcseqs>=2.17.0, legacy 58-slot checkpoint backward compatibility, and 37-slot warm-start.
"""

from __future__ import annotations

import mhcseqs.imgt as mhcseqs_imgt
import pytest
import torch

from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.models.imgt_groove import (
    NUM_CANONICAL_POCKET_SLOTS,
    POCKET_SLOT_SCHEME_LEGACY58,
    POCKET_SLOT_SCHEME_UNION46,
    DirectPocketBilinearHead,
    IMGTPocketCrossInteraction,
    canonical_class_pocket_slot_indices,
    pocket_slot_overlap_mapping,
    pocket_slot_specs,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)

MISSING_9_SLOTS = (
    ("HALF1", "26"),
    ("HALF1", "33"),
    ("HALF1", "56"),
    ("HALF1", "61"),
    ("HALF2", "22"),
    ("HALF2", "59"),
    ("HALF2", "61A"),
    ("HALF2", "72A"),
    ("HALF2", "76"),
)


def test_pocket_slots_match_mhcseqs_217_exactly():
    """Canonical pocket slots match mhcseqs.imgt 3D-verified 46-slot union & class-34 sets."""
    assert NUM_CANONICAL_POCKET_SLOTS == 46
    assert pocket_slot_specs(POCKET_SLOT_SCHEME_UNION46) == tuple(
        mhcseqs_imgt.IMGT_PAN_CLASS_POCKET_SLOTS
    )
    specs_set = set(pocket_slot_specs(POCKET_SLOT_SCHEME_UNION46))
    for slot in MISSING_9_SLOTS:
        assert slot in specs_set, f"missing #165 3D pocket contact {slot}"
    # Bogus mature-numbered entries from legacy58 must NOT be present
    assert ("HALF1", "58") not in specs_set
    assert ("HALF2", "11") not in specs_set

    # Verify exact 37-slot overlap between legacy58 and mhcseqs_union46
    overlap = pocket_slot_overlap_mapping(POCKET_SLOT_SCHEME_LEGACY58, POCKET_SLOT_SCHEME_UNION46)
    assert len(overlap) == 37


def test_class_specific_contact_bias_priors_and_bilinear_shapes():
    """IMGTPocketCrossInteraction uses class-specific 34-slot +2.5 priors; bilinear uses 9x46."""
    ipc = IMGTPocketCrossInteraction(d_model=32, n_heads=4, dropout=0.0)
    assert ipc.n_pocket_slots == 46
    assert ipc.pocket_slot_idx.shape == (46,)

    c1_idx = set(canonical_class_pocket_slot_indices("I", POCKET_SLOT_SCHEME_UNION46))
    c2_idx = set(canonical_class_pocket_slot_indices("II", POCKET_SLOT_SCHEME_UNION46))
    assert len(c1_idx) == 34
    assert len(c2_idx) == 34
    assert len(c1_idx | c2_idx) == 46
    assert len(c1_idx & c2_idx) == 22

    # Class I contact_bias[0] has +2.5 strictly on NetMHCpan-34; Class II on NetMHCIIpan-34
    b0 = ipc.contact_bias[0, 0, 0]
    b1 = ipc.contact_bias[1, 0, 0]
    assert set(torch.nonzero(b0 == 2.5, as_tuple=False).flatten().tolist()) == c1_idx
    assert set(torch.nonzero(b1 == 2.5, as_tuple=False).flatten().tolist()) == c2_idx
    assert (b0 == 0.0).sum().item() == 202 - 34
    assert (b1 == 0.0).sum().item() == 202 - 34

    dpb = DirectPocketBilinearHead(mhc_classes=("I", "II"), d_rank=8, d_hidden=32, dropout=0.0)
    assert dpb.n_pocket_slots == 46
    assert dpb.by_class["I"].mhc_v.shape == (46, 22, 8)
    assert dpb.by_class["II"].mhc_v.shape == (46, 22, 8)
    assert dpb.by_class["I"].linear_energy.in_features == 9 * 46 + 3
    assert dpb.by_class["II"].linear_energy.in_features == 9 * 46 + 3


def _tiny_config(*, scheme: str) -> ModularConfig:
    return ModularConfig(
        endpoints=("ic50",),
        mhc_classes=("I", "II"),
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        imgt_groove_embedding=True,
        imgt_pocket_cross=True,
        direct_pocket_bilinear=True,
        imgt_pocket_slot_scheme=scheme,
    )


def _sample_obs() -> list[Observation]:
    g1_a = (
        "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
    )
    g2_a = (
        "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAA"
        "DMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQR"
    )
    g1_dr = "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITN"
    g2_dr = (
        "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEY"
        "RAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQRR"
    )
    return [
        Observation(
            observation_id="o1",
            endpoint="ic50",
            source="test",
            peptide="NLVPMVATV",
            mhc=(MHCInput(mhc_class="I", groove1=g1_a, groove2=g2_a),),
            context=Context(),
        ),
        Observation(
            observation_id="o2",
            endpoint="ic50",
            source="test",
            peptide="PKYVKQNTLKLAT",
            mhc=(MHCInput(mhc_class="II", groove1=g1_dr, groove2=g2_dr),),
            context=Context(),
        ),
    ]


def test_legacy58_checkpoint_autodetects_and_migrates_to_union46(tmp_path):
    """Pre-#165 58-slot checkpoints load strictly and warm-start cleanly into 46-slot union."""
    torch.manual_seed(7)
    old_model = ModularPresto(_tiny_config(scheme=POCKET_SLOT_SCHEME_LEGACY58)).eval()
    # Set distinctive non-zero weights on direct_pocket_bilinear to verify slot transfer
    with torch.no_grad():
        for cls in ("I", "II"):
            blk = old_model.direct_pocket_bilinear.by_class[cls]
            blk.mhc_v.copy_(
                torch.arange(blk.mhc_v.numel(), dtype=torch.float32).view_as(blk.mhc_v) * 1e-3
            )
            blk.mhc_bias.copy_(
                torch.arange(blk.mhc_bias.numel(), dtype=torch.float32).view_as(blk.mhc_bias) * 1e-2
            )
            blk.linear_energy.weight.copy_(
                torch.arange(blk.linear_energy.weight.numel(), dtype=torch.float32).view_as(
                    blk.linear_energy.weight
                )
                * 1e-4
            )
            blk.mlp[0].weight.copy_(
                torch.arange(blk.mlp[0].weight.numel(), dtype=torch.float32).view_as(
                    blk.mlp[0].weight
                )
                * 1e-5
            )

    obs = _sample_obs()
    with torch.no_grad():
        ref_vals = [float(o.value.item()) for o in old_model(obs)]

    ckpt = tmp_path / "legacy58.pt"
    save_checkpoint(ckpt, old_model, {"ic50": TargetScale(0.0, 1.0, 10)})
    # Strip imgt_pocket_slot_scheme to replicate pre-#165 checkpoints verbatim
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["config"].pop("imgt_pocket_slot_scheme", None)
    torch.save(payload, ckpt)

    reloaded, _, _ = load_checkpoint(ckpt, device="cpu")
    assert reloaded.config.imgt_pocket_slot_scheme == POCKET_SLOT_SCHEME_LEGACY58
    assert reloaded.imgt_pocket_cross.n_pocket_slots == 58
    assert reloaded.direct_pocket_bilinear.n_pocket_slots == 58
    with torch.no_grad():
        got_vals = [float(o.value.item()) for o in reloaded.eval()(obs)]
    assert got_vals == pytest.approx(ref_vals, abs=1e-6)

    # Warm-start migration from legacy58 -> mhcseqs_union46 requires allow_interaction_expansion
    new_model = ModularPresto(_tiny_config(scheme=POCKET_SLOT_SCHEME_UNION46))
    with pytest.raises(ValueError, match="allow_interaction_expansion"):
        initialize_from_checkpoint(new_model, ckpt, allow_interaction_expansion=False)

    initialize_from_checkpoint(new_model, ckpt, allow_interaction_expansion=True)
    assert new_model.config.imgt_pocket_slot_scheme == POCKET_SLOT_SCHEME_UNION46
    assert new_model.direct_pocket_bilinear.n_pocket_slots == 46
    # Verify all 37 overlapping IMGT pocket slots transferred verbatim into direct_pocket_bilinear
    overlap = pocket_slot_overlap_mapping(POCKET_SLOT_SCHEME_LEGACY58, POCKET_SLOT_SCHEME_UNION46)
    for cls in ("I", "II"):
        old_blk = old_model.direct_pocket_bilinear.by_class[cls]
        new_blk = new_model.direct_pocket_bilinear.by_class[cls]
        for old_s, new_s in overlap:
            assert torch.allclose(new_blk.mhc_v[new_s], old_blk.mhc_v[old_s])
            assert torch.allclose(new_blk.mhc_bias[new_s], old_blk.mhc_bias[old_s])
            for p in range(9):
                assert torch.allclose(
                    new_blk.linear_energy.weight[:, p * 46 + new_s],
                    old_blk.linear_energy.weight[:, p * 58 + old_s],
                )
        # PFR columns [-3:] preserved verbatim
        assert torch.allclose(
            new_blk.linear_energy.weight[:, -3:], old_blk.linear_energy.weight[:, -3:]
        )
    # direct_mlp final layer kept zero-init so step-0 direct_mlp perturbation is exact 0.0
    assert float(new_model.imgt_pocket_cross.direct_mlp[-1].weight.detach().abs().max()) == 0.0

    # Verify newly recovered pocket slots receive non-zero step-1 gradients
    new_model.train()
    loss = sum(o.value.pow(2) for o in new_model(obs))
    loss.backward()
    new_only_indices = set(range(46)) - {new_s for _, new_s in overlap}
    assert len(new_only_indices) == 9
    for cls in ("I", "II"):
        grad_lin = new_model.direct_pocket_bilinear.by_class[cls].linear_energy.weight.grad
        assert grad_lin is not None
        new_slot_grad = sum(
            float(grad_lin[:, p * 46 + s].abs().sum()) for p in range(9) for s in new_only_indices
        )
        assert new_slot_grad > 0.0

    # Verify a real train_step updates trained_elements without shape mismatch (regression test)
    train_obs = [
        Observation(
            observation_id=o.observation_id,
            endpoint=o.endpoint,
            source=o.source,
            peptide=o.peptide,
            mhc=o.mhc,
            context=o.context,
            value=100.0,
            unit="nM",
            qualifier="=",
            split="train",
        )
        for o in obs
    ]
    opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    step_out = train_step(new_model, opt, train_obs, {"ic50": TargetScale(0.0, 1.0, 10)})
    assert step_out["loss"] >= 0.0
    for name, p in new_model.named_parameters():
        assert new_model.trained_elements[name].shape == p.shape
