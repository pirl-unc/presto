"""Opt-in affinity-first training controls: defaults, units and exact resume."""

from dataclasses import replace
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from presto.data.modular import AA, MHCInput, Observation, TargetScale, observation_target
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    write_observations,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.tests.test_modular_resume import assert_same, interrupt, manifest_fixture
from presto.training import modular_run as runner
from presto.training.modular import batch_loss, observation_loss, train_step
from presto.training.modular_controls import (
    CONTROL_DEFAULTS,
    EMA,
    cell_blocks,
    cell_key,
    learning_rate,
    ranking_loss,
    segment_for,
    spearman_cells,
    spearman_score,
)
from presto.training.modular_sampling import epoch_order

GOLDEN = Path(__file__).parent / "golden_default_training_trace.json"


def peptides(count, seed=7):
    rng, seen = random.Random(seed), []
    while len(seen) < count:
        candidate = "".join(rng.choice(AA) for _ in range(9))
        if candidate not in seen:
            seen.append(candidate)
    return seen


def fixture(tmp_path, **training):
    """16 train rows (two studies, every 4th class II), 12 validation (two studies), 4 test."""
    sequences = peptides(32)
    rows = []
    for i in range(32):
        split = "train" if i < 16 else "validation" if i < 28 else "test"
        local = i if split == "train" else i - 16 if split == "validation" else i - 28
        study = (
            (f"t{local % 2}" if split == "train" else f"v{local % 2}") if split != "test" else "x"
        )
        rows.append(
            Observation(
                f"a{i}",
                "software",
                "ic50",
                sequences[i],
                (MHCInput(AA, AA[::-1], "II" if split == "train" and i % 4 == 3 else "I"),),
                value=10 ** (1 + (i * 7) % 5),
                unit="nM",
                split=split,
                group_id=f"a{i}",
                provenance={"original_measurement_id": f"a{i}", "study_id": study},
            )
        )
    written = write_observations(tmp_path / "rows.jsonl", rows)
    manifest = {
        "schema": MANIFEST_VERSION,
        "model": {"endpoints": ["ic50"], "mhc_classes": ["I", "II"], "dropout": 0.3},
        "training": {
            "batch_size": 4,
            "max_epochs": 4,
            "max_steps": 10,
            "checkpoint_every_steps": 2,
            **training,
        },
        "sources": [
            {
                "path": written["path"],
                "sha256": written["sha256"],
                "source": "software",
                "population": "artificial",
                "label_policy": "software fixtures only",
                "split_policy": "disjoint artificial inputs",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def preflight_error(tmp_path, match, **training):
    with pytest.raises(ValueError, match=match):
        runner.preflight(ObservationIndex(fixture(tmp_path, **training)))


# ---------------------------------------------------------------- defaults unchanged


def test_default_manifest_reproduces_pre_change_training_trace(tmp_path, monkeypatch):
    golden = json.loads(GOLDEN.read_text())
    original = runner.train_step
    trace = []

    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        assert not set(kwargs) - {"identity_rows", "identity_weight"}
        trace.append(
            {
                "rows": [row.observation_id for row in args[2]],
                "identity": [row.observation_id for row in kwargs.get("identity_rows", [])],
                "loss": result["loss"],
                "parts": result["loss_parts"],
            }
        )
        return result

    monkeypatch.setattr(runner, "train_step", step)
    summary = runner.fit_index(ObservationIndex(manifest_fixture(tmp_path)), tmp_path / "run")
    assert [(t["rows"], t["identity"]) for t in trace] == [
        (t["rows"], t["identity"]) for t in golden["trace"]
    ]
    for new, old in zip(trace, golden["trace"], strict=True):
        assert new["parts"] == pytest.approx(old["parts"], rel=1e-5)
        assert new["loss"] == pytest.approx(old["loss"], rel=1e-5)
    # The resolved training dict is unchanged: no control key is added at its default.
    resolved = json.loads((tmp_path / "run" / "resolved-config.json").read_text())["training"]
    assert resolved == golden["training"] and not set(resolved) & set(CONTROL_DEFAULTS)
    assert summary["selected_epoch"] == golden["selected_epoch"]
    state = torch.load(tmp_path / "run" / "selected.pt", weights_only=True)["state_dict"]
    for name, total in golden["param_sum"].items():
        assert float(state[name].double().sum()) == pytest.approx(total, rel=1e-5, abs=1e-6)
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert all(not set(entry) & {"learning_rate", "selection"} for entry in history)
    resume = torch.load(tmp_path / "run" / "resume.pt", weights_only=True)["metadata"]["resume"]
    assert "segment" not in resume["state"] and "ema" not in resume["state"]


# ---------------------------------------------------------------- 1. loss normalization


def endpoint_batch():
    rows = [
        Observation(
            f"r{i}",
            "software",
            "ic50",
            AA[:9],
            (MHCInput(AA, AA[::-1], "I"),),
            value=10 ** (i + 1),
            unit="nM",
            group_id=f"g{i}",
        )
        for i in range(3)
    ]
    rows.append(
        Observation(
            "b0",
            "software",
            "bulk_observation",
            AA[:9],
            (MHCInput(AA, AA[::-1], "I"),),
            value=1,
            unit="binary",
            group_id="gb",
        )
    )
    values = [torch.tensor(0.3), torch.tensor(1.1), torch.tensor(-0.4), torch.tensor(0.7)]
    outputs = [
        SimpleNamespace(observation_id=row.observation_id, endpoint=row.endpoint, value=value)
        for row, value in zip(rows, values, strict=True)
    ]
    return rows, outputs, {"ic50": TargetScale(2.0, 1.0, 2)}


def test_per_endpoint_normalization_stops_row_count_dominance():
    rows, outputs, scales = endpoint_batch()
    model = SimpleNamespace(config=SimpleNamespace(identity_tasks=()))
    losses = [observation_loss(o, r, scales) for r, o in zip(rows, outputs, strict=True)]
    plain, _ = batch_loss(model, rows, outputs, scales)
    assert plain == pytest.approx(float(torch.stack(losses).mean()))
    balanced, counts = batch_loss(model, rows, outputs, scales, normalization="per_endpoint")
    expected = (float(torch.stack(losses[:3]).mean()) + float(losses[3])) / 2
    assert balanced == pytest.approx(expected) and counts == {"ic50": 3, "bulk_observation": 1}
    weighted, _ = batch_loss(
        model,
        rows,
        outputs,
        scales,
        normalization="per_endpoint",
        endpoint_weights={"bulk_observation": 3.0},
    )
    expected = (float(torch.stack(losses[:3]).mean()) + 3 * float(losses[3])) / 4
    assert weighted == pytest.approx(expected)
    only_ic50, _ = batch_loss(model, rows[:3], outputs[:3], scales, normalization="per_endpoint")
    assert only_ic50 == pytest.approx(float(torch.stack(losses[:3]).mean()))


# ---------------------------------------------------------------- 2. schedule


def test_segment_selection():
    schedule = [{"until_step": 3}, {"until_step": 7}]
    assert [segment_for(schedule, s) for s in (0, 2, 3, 6, 7, 50)] == [0, 0, 1, 1, 2, 2]
    assert segment_for([], 99) == 0


SCHEDULE = [
    {
        "until_step": 4,
        "stratum_weights": [
            {"values": ["I"], "weight": 3},
            {"values": ["II"], "weight": 1},
        ],
    }
]
AFTER = [{"values": ["I"], "weight": 1}, {"values": ["II"], "weight": 3}]


def scheduled(tmp_path):
    return fixture(tmp_path, balance_by=["mhc_class"], stratum_weights=AFTER, schedule=SCHEDULE)


def test_schedule_rebuilds_order_at_boundary_from_the_sampler_rng(tmp_path, monkeypatch):
    path = scheduled(tmp_path)
    original, trace = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / "run")
    index = ObservationIndex(path)
    base = index.partitions["train:assay"]
    rng = random.Random(42)
    first = epoch_order(index, base, ["mhc_class"], rng, SCHEDULE[0]["stratum_weights"])
    second = epoch_order(index, base, ["mhc_class"], rng, AFTER)
    assert len(first) == 16 and len(second) == 48  # class II oversampled after the boundary
    expected = [first[i : i + 4] for i in range(0, 16, 4)]
    for start in range(0, len(second), 4):
        expected.append(second[start : start + 4])
    got = [[index[i].observation_id for i in batch] for batch in expected[:10]]
    assert [rows for rows, _ in trace] == got
    state = torch.load(tmp_path / "run" / "resume.pt", weights_only=True)["metadata"]["resume"][
        "state"
    ]
    assert state["segment"] == 1


@pytest.mark.parametrize("before_step", [3, 5, 7])
def test_schedule_resume_is_exact_inside_and_across_a_boundary(tmp_path, monkeypatch, before_step):
    path = scheduled(tmp_path)
    original, trace = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / "reference")
    expected = list(trace)
    monkeypatch.setattr(runner, "train_step", original)
    original, before = interrupt(monkeypatch, before_step=before_step)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), tmp_path / "resumed")
    snapshot = torch.load(tmp_path / "resumed" / "resume.pt", weights_only=True)
    done = snapshot["metadata"]["resume"]["state"]["steps"]
    if before_step == 5:  # snapshot taken on the boundary step, rebuild still pending
        assert done == 4 and snapshot["metadata"]["resume"]["state"]["segment"] == 0
    monkeypatch.setattr(runner, "train_step", original)
    original, after = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / "resumed", resume=True)
    assert before[:done] + after == expected
    for name in ("selected.pt", "resume.pt", "last.pt"):
        a = torch.load(tmp_path / "reference" / name, weights_only=True)
        b = torch.load(tmp_path / "resumed" / name, weights_only=True)
        assert_same(a["state_dict"], b["state_dict"])
        assert_same(a["optimizer"], b["optimizer"])


# ---------------------------------------------------------------- 3. LR, clip, EMA


def test_learning_rate_schedule_shape():
    cosine = {"warmup_steps": 4, "decay": "cosine", "min_lr_fraction": 0.1}
    rates = [learning_rate(1.0, cosine, s, 14) for s in range(15)]
    assert rates[:4] == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert rates[4] == pytest.approx(1.0) and rates[14] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(rates[4:], rates[5:]))
    assert learning_rate(1.0, {"warmup_steps": 0, "decay": "none"}, 99, None) == 1.0
    assert learning_rate(0.5, None, 3, None) == 0.5
    assert learning_rate(1.0, {**cosine, "decay_steps": 24}, 14, None) == pytest.approx(0.55)


def tiny_model():
    torch.manual_seed(3)
    return ModularPresto(ModularConfig(endpoints=("ic50",), mhc_classes=("I",), dropout=0))


def test_grad_clip_bounds_the_applied_gradient_norm():
    row = Observation(
        "r",
        "software",
        "ic50",
        AA[:9],
        (MHCInput(AA, AA[::-1], "I"),),
        value=1e6,
        unit="nM",
        group_id="g",
    )
    scales = {"ic50": TargetScale(0.0, 0.1, 2)}
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    free = train_step(model, optimizer, [row], scales)
    assert "grad_norm" not in free
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clipped = train_step(model, optimizer, [row], scales, grad_clip_norm=0.05)
    norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
    assert clipped["grad_norm"] > 0.05 and float(norm) == pytest.approx(0.05, rel=1e-4)


def test_ema_update_swap_and_restore_are_exact():
    model = tiny_model()
    ema = EMA(model, 0.9)
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    decay = min(0.9, 2 / 11)
    name = ema.names[0]
    assert torch.allclose(ema.shadow[name], decay * start[name] + (1 - decay) * (start[name] + 1))
    raw = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.swap(model):
        assert torch.equal(dict(model.named_parameters())[name], ema.shadow[name])
    assert_same(raw, {n: p.detach() for n, p in model.named_parameters()})
    clone = EMA(model, 0.9, saved=ema.state())
    assert clone.updates == 1 and clone.names == ema.names
    with pytest.raises(ValueError, match="EMA"):
        EMA(model, 0.9, saved={"updates": 1, "shadow": {"x": torch.zeros(1)}})


OPTIMIZATION = dict(
    lr_schedule={"warmup_steps": 2, "decay": "cosine", "min_lr_fraction": 0.1},
    grad_clip_norm=0.5,
    ema_decay=0.9,
    loss_normalization="per_endpoint",
)


def run_with_interruption(tmp_path, monkeypatch, path, before_step):
    original, trace = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / "reference")
    expected = list(trace)
    monkeypatch.setattr(runner, "train_step", original)
    original, before = interrupt(monkeypatch, before_step=before_step)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.fit_index(ObservationIndex(path), tmp_path / "resumed")
    done = torch.load(tmp_path / "resumed" / "resume.pt", weights_only=True)["metadata"]["resume"][
        "state"
    ]["steps"]
    monkeypatch.setattr(runner, "train_step", original)
    original, after = interrupt(monkeypatch, before_step=100)
    runner.fit_index(ObservationIndex(path), tmp_path / "resumed", resume=True)
    assert before[:done] + after == expected
    for name in ("selected.pt", "resume.pt", "last.pt"):
        a = torch.load(tmp_path / "reference" / name, weights_only=True)
        b = torch.load(tmp_path / "resumed" / name, weights_only=True)
        assert_same(a["state_dict"], b["state_dict"])
        assert_same(a["optimizer"], b["optimizer"])
        assert_same(
            a["metadata"].get("resume", {}).get("state", {}).get("ema"),
            b["metadata"].get("resume", {}).get("state", {}).get("ema"),
        )
    return tmp_path / "reference"


def test_lr_clip_ema_resume_is_exact_and_selection_uses_ema_weights(tmp_path, monkeypatch):
    path = fixture(tmp_path, **OPTIMIZATION)
    reference = run_with_interruption(tmp_path, monkeypatch, path, before_step=6)
    selected = torch.load(reference / "selected.pt", weights_only=True)
    resume = torch.load(reference / "resume.pt", weights_only=True)
    assert selected["metadata"]["weights"] == "ema"
    shadow = resume["metadata"]["resume"]["state"]["ema"]["shadow"]
    last = torch.load(reference / "last.pt", weights_only=True)
    for name, value in shadow.items():
        assert torch.equal(last["state_dict"][name], value)  # last.pt saved the EMA
        assert not torch.equal(resume["state_dict"][name], value)  # resume keeps raw weights
    history = json.loads((reference / "history.json").read_text())
    assert history[0]["learning_rate"] > history[-1]["learning_rate"] > 0
    assert history[-1]["learning_rate"] == pytest.approx(
        learning_rate(3e-4, OPTIMIZATION["lr_schedule"], 9, 10)
    )


# ---------------------------------------------------------------- 4. ranking + cell batching


def cell_rows():
    def make(i, study, value, qualifier="="):
        return Observation(
            f"c{i}",
            "software",
            "ic50",
            AA[:9],
            (MHCInput(AA, AA[::-1], "I"),),
            value=value,
            unit="nM",
            qualifier=qualifier,
            group_id=f"g{i}",
            provenance={"study_id": study} if study else {},
        )

    return [
        make(0, "S1", 10),
        make(1, "S1", 100),
        make(2, "S1", 1000),
        make(3, "S1", 5000, ">"),  # censored rows never pair
        make(4, "S2", 10),
        make(5, "S2", 1000),
        make(6, None, 10),
        make(7, None, 1000),  # no study: no cell
    ]


def test_ranking_loss_pairs_only_exact_same_cell_rows():
    rows = cell_rows()
    scores = [
        torch.tensor(v, requires_grad=True) for v in (2.0, 0.5, -1.0, 0.0, 0.1, 0.9, 3.0, -3.0)
    ]
    outputs = [SimpleNamespace(value=s) for s in scores]
    loss, pairs = ranking_loss(rows, outputs, ["ic50"])
    # S1: (10<100),(10<1000),(100<1000) with scores 2,.5,-1 (all inverted); S2: 10<1000 (.1<.9)
    expected = [2 - 0.5, 2 + 1, 0.5 + 1, 0.1 - 0.9]
    assert pairs == 4
    assert float(loss.detach()) == pytest.approx(float(F.softplus(torch.tensor(expected)).mean()))
    loss.backward()
    assert scores[3].grad is None and scores[6].grad is None and scores[7].grad is None
    assert scores[0].grad > 0 and scores[2].grad < 0
    assert ranking_loss(rows, outputs, ["ec50"]) == (None, 0)
    assert cell_key(rows[0]) != cell_key(rows[4]) and cell_key(rows[6]) is None


def test_cell_blocks_keep_the_row_multiset_and_group_same_cell_rows():
    rows = cell_rows() * 2
    keys = {i: cell_key(row) for i, row in enumerate(rows)}
    order = list(range(len(rows)))
    first = cell_blocks(order, keys, 2, random.Random(1))
    assert sorted(first) == order
    assert first == cell_blocks(order, keys, 2, random.Random(1))
    assert first != cell_blocks(order, keys, 2, random.Random(2))
    blocks = [first[i : i + 2] for i in range(0, len(first), 2)]
    assert all(len({keys[i] for i in block}) == 1 for block in blocks)


def test_ranking_run_cobatches_cells_and_resumes_exactly(tmp_path, monkeypatch):
    path = fixture(
        tmp_path,
        within_group_ranking_loss={"weight": 0.5, "endpoints": ["ic50"]},
        cell_batching={"rows_per_cell": 4},
    )
    seen = []
    original = runner.train_step

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result["loss_parts"])
        return result

    monkeypatch.setattr(runner, "train_step", spy)
    reference = run_with_interruption(tmp_path, monkeypatch, path, before_step=6)
    assert any(parts.get("ranking_pairs", 0) > 0 and "ranking" in parts for parts in seen)
    index = ObservationIndex(path)
    for step_rows in json.loads((reference / "history.json").read_text()):
        assert step_rows["steps"] > 0
    # Each batch of four draws one study cell, so every pair is admissible.
    monkeypatch.setattr(runner, "train_step", original)
    batches = []

    def capture(*args, **kwargs):
        batches.append({cell_key(row) for row in args[2]})
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "train_step", capture)
    runner.fit_index(ObservationIndex(path), tmp_path / "again")
    assert batches and all(len(cells) == 1 for cells in batches)
    assert index.partitions["train:assay"]


# ---------------------------------------------------------------- 5. Spearman selection


def record(endpoint, study, target, prediction, qualifier="="):
    return {
        "endpoint": endpoint,
        "qualifier": qualifier,
        "target": target,
        "prediction": prediction,
        "observation": {"provenance": {"study_id": study} if study else {}},
    }


def test_spearman_cells_skip_small_studies_and_exclude_censored_rows():
    records = [record("ic50", "A", i, i * 2.0) for i in range(5)]  # +1
    records += [record("ic50", "B", i, -i) for i in range(6)]  # -1
    records += [record("ic50", "C", i, 1.0) for i in range(5)]  # constant: 0
    records += [record("ic50", "D", i, i) for i in range(3)]  # too small
    records += [record("ic50", "A", 9, -50.0, qualifier=">")]  # censored, ignored
    records += [record("ic50", None, i, i) for i in range(9)]  # no study
    records += [record("ec50", "A", i, i) for i in range(9)]  # undeclared endpoint
    used, skipped = spearman_cells(records, ["ic50"], 5)
    assert {(c["study"], c["n"]) for c in used} == {("A", 5), ("B", 6), ("C", 5)}
    assert {c["study"]: c["spearman"] for c in used} == pytest.approx({"A": 1, "B": -1, "C": 0})
    assert skipped == [{"endpoint": "ic50", "study": "D", "n": 3}]
    score, report = spearman_score(records, ["ic50"], 5)
    assert score == pytest.approx(0.0) and len(report["cells"]) == 3
    with pytest.raises(ValueError, match="no validation cell"):
        spearman_score(records, ["ic50"], 50)


def test_spearman_selection_run_reports_cells_and_selects_on_it(tmp_path):
    path = fixture(
        tmp_path,
        selection_metric="within_study_spearman",
        selection_spearman_endpoints=["ic50"],
        selection_min_cell_rows=5,
    )
    _, _, _, admission = runner.preflight(ObservationIndex(path))
    assert [(c["study"], c["n"]) for c in admission["selection_cells"]] == [("v0", 6), ("v1", 6)]
    runner.fit_index(ObservationIndex(path), tmp_path / "run")
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert all({c["study"] for c in e["selection"]["cells"]} == {"v0", "v1"} for e in history)
    selected = torch.load(tmp_path / "run" / "selected.pt", weights_only=True)["metadata"]
    scores = [-sum(c["spearman"] for c in e["selection"]["cells"]) / 2 for e in history]
    assert selected["selection_metric"] == pytest.approx(min(scores))


# ---------------------------------------------------------------- preflight validation


@pytest.mark.parametrize(
    "training, match",
    [
        ({"loss_normalization": "mean"}, "loss_normalization"),
        ({"endpoint_loss_weights": {"ic50": 2}}, "per_endpoint"),
        ({"loss_normalization": "per_endpoint", "endpoint_loss_weights": {"tm": 1}}, "active"),
        ({"loss_normalization": "per_endpoint", "endpoint_loss_weights": {"ic50": 0}}, "positive"),
        ({"schedule": SCHEDULE}, "balance_by"),
        (
            {"balance_by": ["mhc_class"], "schedule": [{"until_step": 3}]},
            "until_step",
        ),
        (
            {"balance_by": ["mhc_class"], "schedule": SCHEDULE + [dict(SCHEDULE[0])]},
            "increasing",
        ),
        (
            {
                "balance_by": ["mhc_class"],
                "schedule": [
                    {"until_step": 2, "stratum_weights": [{"values": ["I"], "weight": 1}]}
                ],
            },
            "every observed",
        ),
        ({"lr_schedule": {"decay": "linear"}}, "decay"),
        ({"lr_schedule": {"decay": "cosine"}, "max_steps": None}, "decay_steps or max_steps"),
        ({"lr_schedule": {"min_lr_fraction": 2}}, "min_lr_fraction"),
        ({"lr_schedule": {"warmup_steps": 5, "decay_steps": 5}}, "decay_steps"),
        ({"lr_schedule": {"bogus": 1}}, "lr_schedule"),
        ({"grad_clip_norm": 0}, "grad_clip_norm"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"within_group_ranking_loss": {"weight": 1, "endpoints": []}}, "endpoints"),
        ({"within_group_ranking_loss": {"weight": -1, "endpoints": ["ic50"]}}, "weight"),
        ({"within_group_ranking_loss": {"weight": 1, "endpoints": ["bulk_observation"]}}, "active"),
        ({"cell_batching": {"rows_per_cell": 1}}, "rows_per_cell"),
        ({"cell_batching": {"rows_per_cell": 99}}, "rows_per_cell"),
        ({"selection_metric": "auroc"}, "selection_metric"),
        ({"selection_metric": "within_study_spearman"}, "selection_spearman_endpoints"),
        ({"selection_spearman_endpoints": ["ic50"]}, "requires selection_metric"),
        (
            {
                "selection_metric": "within_study_spearman",
                "selection_spearman_endpoints": ["ic50"],
                "selection_endpoint": "ic50",
            },
            "cannot be combined",
        ),
        (
            {
                "selection_metric": "within_study_spearman",
                "selection_spearman_endpoints": ["ic50"],
                "selection_min_cell_rows": 7,
            },
            "selection_min_cell_rows",
        ),
        ({"selection_min_cell_rows": 1}, "selection_min_cell_rows"),
    ],
)
def test_preflight_rejects_invalid_controls(tmp_path, training, match):
    preflight_error(tmp_path, match, **training)


def test_valid_controls_enter_the_resume_contract(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = runner.preflight(ObservationIndex(fixture(tmp_path / "a")))[1]
    chosen = runner.preflight(ObservationIndex(fixture(tmp_path / "b", ema_decay=0.99)))[1]
    assert not set(base) & set(CONTROL_DEFAULTS) and chosen["ema_decay"] == 0.99
    assert base != chosen
    assert replace  # keep the import used by fixtures below
    assert math.isfinite(observation_target(cell_rows()[0]))
