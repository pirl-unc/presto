"""Search specificity and exact legacy-checkpoint behavior (Presto #82)."""

import random

import pytest
import torch

from presto.data.bulk_ms import _mismatched_machinery, would_cleave
from presto.data.vocab import AA_TO_IDX, EXCISION_MACHINERY_TO_IDX
from presto.models.presto import Presto
from presto.training.checkpointing import load_model_from_checkpoint, save_model_checkpoint


@pytest.mark.parametrize("residue", ["K", "R"])
def test_trypsin_p_allows_proline_and_terminal_cleavage(residue):
    assert would_cleave("trypsin_p", residue, "P")
    assert not would_cleave("trypsin", residue, "P")
    for enzyme in ("trypsin", "trypsin_p"):
        assert would_cleave(enzyme, residue, "")
        assert would_cleave(enzyme, residue, "A")
        assert not would_cleave(enzyme, "A", "P")


def test_generated_controls_never_mislabel_trypsin_p_at_kp():
    choices = {
        _mismatched_machinery("ACDEFGHIK", "PAA", "lysc", random.Random(seed))
        for seed in range(100)
    }
    assert "trypsin" in choices
    assert "trypsin_p" not in choices
    assert all(not would_cleave(enzyme, "K", "P") for enzyme in choices)


def _inputs():
    def encode(sequence):
        return torch.tensor([[AA_TO_IDX[aa] for aa in sequence]])

    return {
        "pep_tok": encode("ACDEFGHIK"),
        "mhc_a_tok": encode("ACDEFGHIKLMNPQRSTVWY"),
        "mhc_b_tok": encode("ACDEFGHIKLMNPQRSTVWY"),
        "flank_n_tok": encode("AAA"),
        "flank_c_tok": encode("PAA"),
        "mhc_class": "I",
    }


def test_trypsin_p_model_route_and_junction_rule():
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    with torch.no_grad():
        output = model(**_inputs(), machinery="trypsin_p")
    assert output["excision_machinery_idx"].item() == EXCISION_MACHINERY_TO_IDX["trypsin_p"]
    head = model.excision_head
    # Isolate the pinned junction rule from changes in the learned processing context.
    p1 = torch.tensor([AA_TO_IDX["K"], AA_TO_IDX["R"]])
    before_p = torch.full_like(p1, AA_TO_IDX["P"])
    before_a = torch.full_like(p1, AA_TO_IDX["A"])
    scores = {}
    for enzyme in ("trypsin", "trypsin_p"):
        machinery = torch.full_like(p1, EXCISION_MACHINERY_TO_IDX[enzyme])
        scores[enzyme] = [
            head._junction_score(
                head.effective_profile_c(),
                head.profile_scale_c,
                p1,
                next_residue,
                machinery,
                torch.zeros_like(p1, dtype=torch.float),
            )
            for next_residue in (before_p, before_a)
        ]
    assert torch.equal(*scores["trypsin_p"])
    assert torch.all(scores["trypsin"][0] < scores["trypsin"][1])


@pytest.mark.parametrize("version", [1, 2])
def test_checkpoint_roundtrip_preserves_all_weights_and_predictions(tmp_path, version):
    model = Presto(d_model=32, n_layers=1, n_heads=4, excision_machinery_version=version).eval()
    path = tmp_path / "model.pt"
    payload = save_model_checkpoint(path, model=model)
    assert payload["model_config"]["excision_machinery_version"] == version
    if version == 1:
        # Historical payloads have no vocabulary version, but have the exact seven-entry head.
        del payload["model_config"]["excision_machinery_version"]
        torch.save(payload, path)
    loaded, _ = load_model_from_checkpoint(path)
    loaded.eval()
    assert loaded.excision_machinery_version == version
    assert len(loaded.excision_machinery_vocabulary) == (7 if version == 1 else 8)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    with torch.no_grad():
        expected = model(**_inputs(), machinery="trypsin")
        actual = loaded(**_inputs(), machinery="trypsin")
    for key in ("excision_logit", "excision_c_terminus_score", "ms_detectability_logit"):
        assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize(
    "machinery", ["trypsin_p", ["trypsin_p"], torch.tensor([7]), torch.tensor([-1])]
)
def test_legacy_model_rejects_unavailable_category(machinery):
    model = Presto(d_model=32, n_layers=1, n_heads=4, excision_machinery_version=1)
    with pytest.raises(ValueError, match="unavailable in vocabulary version 1"):
        model(**_inputs(), machinery=machinery)


@pytest.mark.parametrize("version", [0, 3, "2", True])
def test_unknown_vocabulary_version_is_rejected(version):
    with pytest.raises(ValueError, match="vocabulary version"):
        Presto(d_model=32, n_layers=1, n_heads=4, excision_machinery_version=version)


@pytest.mark.parametrize("declared", [1, 99])
def test_checkpoint_cannot_lie_about_its_head_version(tmp_path, declared):
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    path = tmp_path / "model.pt"
    payload = save_model_checkpoint(path, model=model)
    payload["model_config"]["excision_machinery_version"] = declared
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_model_from_checkpoint(path, strict=False)


def test_unrecognized_head_shape_is_not_guessed(tmp_path):
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    path = tmp_path / "model.pt"
    payload = save_model_checkpoint(path, model=model)
    payload["model_state_dict"]["excision_head.bias"] = torch.zeros(9)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="head shape"):
        load_model_from_checkpoint(path)


def test_bare_legacy_state_dict_recovers_vocabulary(tmp_path):
    model = Presto(d_model=32, n_layers=1, n_heads=4, excision_machinery_version=1)
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    restored, _ = load_model_from_checkpoint(path, n_heads=4)
    assert restored.excision_machinery_version == 1
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )


@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf")])
def test_fractional_or_nonfinite_machinery_index_is_rejected(value):
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    with pytest.raises(ValueError, match="finite integers"):
        model(**_inputs(), machinery=torch.tensor([value]))


def test_trainer_rejects_legacy_resume_before_mutating_model(tmp_path):
    from presto.training.trainer import Trainer

    legacy = Trainer(Presto(d_model=32, n_layers=1, n_heads=4, excision_machinery_version=1))
    path = tmp_path / "legacy.pt"
    legacy.save_checkpoint(path)
    current = Trainer(Presto(d_model=32, n_layers=1, n_heads=4))
    before = {key: tensor.clone() for key, tensor in current.model.state_dict().items()}
    with pytest.raises(ValueError, match="does not match trainer model"):
        current.load_checkpoint(path)
    assert all(
        torch.equal(before[key], tensor) for key, tensor in current.model.state_dict().items()
    )
    restored, _ = load_model_from_checkpoint(path)
    compatible = Trainer(restored)
    compatible.load_checkpoint(path)
    assert compatible.model.excision_machinery_version == 1
