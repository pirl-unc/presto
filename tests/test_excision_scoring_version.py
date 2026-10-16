"""Known-boundary regressions for versioned internal cleavage scoring (#88)."""

import math

import pytest
import torch

from presto.data.vocab import AA_TO_IDX, EXCISION_MACHINERY_TO_IDX
from presto.models.presto import Presto
from presto.training.checkpointing import load_model_from_checkpoint, save_model_checkpoint


@pytest.fixture(scope="module")
def heads():
    return {
        version: Presto(
            d_model=32, n_layers=1, n_heads=4, excision_scoring_version=version
        ).excision_head
        for version in (1, 2)
    }


def score(head, sequence, enzyme):
    # Underscores are padding, not amino-acid symbols.
    tokens = torch.tensor(
        [[0 if aa == "_" else AA_TO_IDX[aa] for aa in sequence]], dtype=torch.long
    )
    machinery = torch.tensor([EXCISION_MACHINERY_TO_IDX[enzyme]])
    return head._missed_cleavage_score(tokens, machinery).item()


@pytest.mark.parametrize("sequence", ["AKPRA", "ARPKA"])
def test_internal_proline_blocks_only_ordinary_trypsin(heads, sequence):
    assert score(heads[2], sequence, "trypsin") == pytest.approx(-math.log(2))
    assert score(heads[2], sequence, "trypsin_p") == pytest.approx(-2 * math.log(2))
    for enzyme in ("trypsin", "trypsin_p"):
        assert score(heads[1], sequence, enzyme) == pytest.approx(-2 * math.log(2))


@pytest.mark.parametrize("sequence", ["AKARA", "__AKARA", "AKARA___", "__AKARA___"])
def test_corrected_internal_sites_tolerate_padding(heads, sequence):
    assert score(heads[2], sequence, "trypsin") == pytest.approx(-2 * math.log(2))


@pytest.mark.parametrize("sequence", ["AK", "__AK__", "R", "", "___", "AK__PA"])
def test_terminal_or_missing_next_residue_is_not_an_internal_site(heads, sequence):
    assert score(heads[2], sequence, "trypsin") == 0


@pytest.mark.parametrize("enzyme,sequence", [("chymotrypsin", "AFPA"), ("gluc", "AEPA")])
def test_other_declared_proline_exclusions_apply(heads, enzyme, sequence):
    assert score(heads[2], sequence, enzyme) == 0
    assert score(heads[1], sequence, enzyme) == pytest.approx(-math.log(2))


def test_lysc_and_processive_machinery_are_unchanged(heads):
    for version in (1, 2):
        assert score(heads[version], "AKPA", "lysc") == pytest.approx(-math.log(2))
        assert score(heads[version], "AKPA", "proteasome") == 0


@pytest.mark.parametrize("machinery_version", [1, 2])
@pytest.mark.parametrize("scoring_version", [1, 2])
def test_checkpoint_records_or_recovers_scoring_without_changing_predictions(
    tmp_path, machinery_version, scoring_version
):
    model = Presto(
        d_model=32,
        n_layers=1,
        n_heads=4,
        excision_machinery_version=machinery_version,
        excision_scoring_version=scoring_version,
    ).eval()
    path = tmp_path / "model.pt"
    payload = save_model_checkpoint(path, model=model)
    assert payload["model_config"]["excision_scoring_version"] == scoring_version
    if scoring_version == 1:
        del payload["model_config"]["excision_scoring_version"]
        torch.save(payload, path)
    loaded, _ = load_model_from_checkpoint(path)
    loaded.eval()
    assert loaded.excision_scoring_version == scoring_version
    assert loaded.excision_head.scoring_version == scoring_version
    inputs = {
        "pep_tok": torch.tensor([[AA_TO_IDX[aa] for aa in "AAKPRPK"]]),
        "mhc_a_tok": torch.tensor([[AA_TO_IDX["A"]] * 20]),
        "mhc_b_tok": torch.tensor([[AA_TO_IDX["A"]] * 20]),
        "machinery": "trypsin",
        "mhc_class": "I",
    }
    with torch.no_grad():
        expected, actual = model(**inputs), loaded(**inputs)
    for key in ("excision_missed_cleavage_score", "excision_logit", "ms_detectability_logit"):
        assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize("version", [0, 3, "2", True])
def test_unknown_scoring_version_fails(version, tmp_path):
    with pytest.raises(ValueError, match="scoring version"):
        Presto(d_model=32, n_layers=1, n_heads=4, excision_scoring_version=version)
    path = tmp_path / "bad.pt"
    payload = save_model_checkpoint(path, model=Presto(d_model=32, n_layers=1, n_heads=4))
    payload["model_config"]["excision_scoring_version"] = version
    torch.save(payload, path)
    with pytest.raises(ValueError, match="scoring version"):
        load_model_from_checkpoint(path)


def test_trainer_semantic_mismatch_fails_before_weights_are_loaded(tmp_path):
    from presto.training.trainer import Trainer

    path = tmp_path / "old.pt"
    legacy = Trainer(Presto(d_model=32, n_layers=1, n_heads=4, excision_scoring_version=1))
    legacy.save_checkpoint(path)
    payload = torch.load(path)
    del payload["model_config"]["excision_scoring_version"]
    torch.save(payload, path)
    current = Trainer(Presto(d_model=32, n_layers=1, n_heads=4))
    before = {key: value.clone() for key, value in current.model.state_dict().items()}
    with pytest.raises(ValueError, match="scoring version 1 does not match"):
        current.load_checkpoint(path)
    assert all(torch.equal(before[key], value) for key, value in current.model.state_dict().items())
    loaded, _ = load_model_from_checkpoint(path)
    compatible = Trainer(loaded)
    compatible.load_checkpoint(path)
    assert compatible.model.excision_scoring_version == 1
