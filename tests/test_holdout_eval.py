"""Tests for generic per-task held-out metrics."""

import numpy as np
import pytest

from presto.training.holdout_eval import (
    TaskPredictionAccumulator,
    auprc,
    auroc,
    binary_metrics,
    flatten_summary,
    metrics_for_loss_type,
    pearson,
    regression_metrics,
    spearman,
    summarize_accumulators,
)


class TestEstimators:
    def test_auroc_perfect_separation(self):
        y = np.array([0, 0, 1, 1.0])
        assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_auroc_inverted(self):
        y = np.array([0, 0, 1, 1.0])
        assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)

    def test_auroc_ties_give_one_half(self):
        y = np.array([0, 1.0])
        assert auroc(y, np.array([0.5, 0.5])) == pytest.approx(0.5)

    def test_auroc_undefined_without_both_classes(self):
        assert auroc(np.array([1.0, 1.0]), np.array([0.2, 0.8])) is None

    def test_auprc_perfect(self):
        y = np.array([0, 0, 1, 1.0])
        assert auprc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_auprc_matches_hand_computed_average_precision(self):
        # ranked: pos, neg, pos -> precisions at hits are 1/1 and 2/3
        y = np.array([1.0, 0.0, 1.0])
        score = np.array([0.9, 0.8, 0.7])
        assert auprc(y, score) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)

    def test_spearman_is_rank_based(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        # monotone but very non-linear: Spearman 1, Pearson < 1
        pred = np.array([1.0, 2.0, 4.0, 1000.0])
        assert spearman(y, pred) == pytest.approx(1.0)
        assert pearson(y, pred) < 0.95

    def test_spearman_handles_ties(self):
        y = np.array([1.0, 1.0, 2.0, 2.0])
        assert spearman(y, np.array([5.0, 5.0, 9.0, 9.0])) == pytest.approx(1.0)

    def test_constant_prediction_has_no_correlation(self):
        assert pearson(np.array([1.0, 2.0, 3.0]), np.array([7.0, 7.0, 7.0])) is None


class TestMetricFamilySelection:
    def test_bce_gets_binary_metrics_from_logits(self):
        # logits: negative -> class 0, positive -> class 1
        metrics = metrics_for_loss_type(
            "bce", np.array([0.0, 0.0, 1.0, 1.0]), np.array([-4.0, -3.0, 3.0, 4.0])
        )
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert "auprc" in metrics and "balanced_accuracy" in metrics

    def test_mse_gets_regression_metrics(self):
        metrics = metrics_for_loss_type("mse", np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 2.9]))
        assert "spearman" in metrics and "rmse" in metrics
        assert "auroc" not in metrics

    def test_censor_is_treated_as_regression(self):
        metrics = metrics_for_loss_type(
            "censor", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        )
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_empty_input_returns_no_metrics(self):
        assert metrics_for_loss_type("bce", np.array([]), np.array([])) == {}


class TestAccumulator:
    def test_mask_excludes_unlabeled_rows(self):
        acc = TaskPredictionAccumulator("excision", "bce")
        acc.add(
            y_true=[1.0, 0.0, 1.0],
            y_pred=[5.0, -5.0, 5.0],
            mask=[1.0, 0.0, 1.0],
            sample_ids=["a", "b", "c"],
        )
        assert len(acc) == 2
        assert [row["sample_id"] for row in acc.rows()] == ["a", "c"]

    def test_metrics_and_rows_agree_on_count(self):
        acc = TaskPredictionAccumulator("ms_detectability", "bce")
        acc.add([1.0, 0.0], [2.0, -2.0], [1.0, 1.0])
        assert acc.metrics()["n"] == pytest.approx(2.0)
        assert len(acc.rows()) == 2

    def test_summary_skips_tasks_with_no_examples(self):
        empty = TaskPredictionAccumulator("processing", "bce")
        filled = TaskPredictionAccumulator("excision", "bce")
        filled.add([1.0, 0.0], [2.0, -2.0], [1.0, 1.0])
        summary = summarize_accumulators([empty, filled])
        assert set(summary) == {"excision"}

    def test_flatten_summary_prefixes_task_name(self):
        flat = flatten_summary({"excision": {"auroc": 0.9, "n": 10}})
        assert flat == {"excision_auroc": 0.9, "excision_n": 10.0}


def test_binary_metrics_balanced_accuracy_on_imbalanced_data():
    """A majority-class predictor should score ~0.5 balanced accuracy."""
    y = np.array([0.0] * 90 + [1.0] * 10)
    always_negative = np.zeros(100)
    metrics = binary_metrics(y, always_negative)
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_regression_metrics_rmse_is_exact():
    metrics = regression_metrics(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert metrics["rmse"] == pytest.approx(np.sqrt(12.5))


class TestCollectionAndArtifacts:
    """The collection path, exercised with fakes rather than a real trainer."""

    class _Spec:
        def __init__(self, name, loss_type, pred_paths):
            self.name = name
            self.loss_type = loss_type
            self.pred_paths = pred_paths

    class _Model:
        """Minimal stand-in: the collector only calls .eval() on the model."""

        def eval(self):
            return self

    class _Batch:
        def __init__(self, targets, masks, sample_ids):
            self.targets = targets
            self.masks = masks
            self.sample_ids = sample_ids

        def to(self, _device):
            return self

    def _pieces(self):
        import torch

        spec = self._Spec("excision", "bce", (("excision_logit",),))
        batch = self._Batch(
            targets={"excision": torch.tensor([1.0, 0.0, 1.0])},
            masks={"excision": torch.tensor([1.0, 1.0, 0.0])},
            sample_ids=["s0", "s1", "s2"],
        )
        outputs = {"excision_logit": torch.tensor([3.0, -3.0, 0.0])}
        return spec, batch, outputs

    def test_collect_respects_mask_and_sample_ids(self):
        from presto.training.holdout_eval import collect_holdout_predictions

        spec, batch, outputs = self._pieces()
        accumulators = collect_holdout_predictions(
            model=self._Model(),
            loader=[batch],
            device="cpu",
            specs=[spec],
            forward_fn=lambda _m, _b: outputs,
            resolve_pred_fn=lambda out, paths: out[paths[0][0]],
            get_target_fn=lambda b, s: b.targets[s.name],
            get_mask_fn=lambda b, s: b.masks[s.name],
        )
        accumulator = accumulators["excision"]
        assert len(accumulator) == 2  # third row masked out
        assert [row["sample_id"] for row in accumulator.rows()] == ["s0", "s1"]
        assert accumulator.metrics()["auroc"] == pytest.approx(1.0)

    def test_shape_mismatch_is_skipped_not_crashed(self):
        """Multi-class heads do not line up element-wise with their targets."""
        import torch

        from presto.training.holdout_eval import collect_holdout_predictions

        spec = self._Spec("mhc_class", "ce", (("mhc_class_logits",),))
        batch = self._Batch(
            targets={"mhc_class": torch.tensor([0.0, 1.0])},
            masks={"mhc_class": torch.tensor([1.0, 1.0])},
            sample_ids=["a", "b"],
        )
        outputs = {"mhc_class_logits": torch.zeros(2, 5)}
        accumulators = collect_holdout_predictions(
            model=self._Model(),
            loader=[batch],
            device="cpu",
            specs=[spec],
            forward_fn=lambda _m, _b: outputs,
            resolve_pred_fn=lambda out, paths: out[paths[0][0]],
            get_target_fn=lambda b, s: b.targets[s.name],
            get_mask_fn=lambda b, s: b.masks[s.name],
        )
        assert len(accumulators["mhc_class"]) == 0

    def test_artifacts_are_written(self, tmp_path):
        import csv
        import json

        from presto.training.holdout_eval import (
            TaskPredictionAccumulator,
            write_holdout_artifacts,
        )

        accumulator = TaskPredictionAccumulator("excision", "bce")
        accumulator.add([1.0, 0.0], [4.0, -4.0], [1.0, 1.0], ["s0", "s1"])
        payload = write_holdout_artifacts(
            tmp_path,
            {"excision": accumulator},
            split="val",
            extra_summary={"best_val_loss": 0.25},
        )
        assert payload["best_val_loss"] == 0.25
        assert payload["tasks"]["excision"]["auroc"] == pytest.approx(1.0)

        written = json.loads((tmp_path / "summary.json").read_text())
        assert written["split"] == "val"

        with (tmp_path / "val_predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["sample_id"] for row in rows] == ["s0", "s1"]

    def test_flat_metric_csv_is_written(self, tmp_path):
        """Written directly, not via RunLogger, which the trainer closes first."""
        import csv

        from presto.training.holdout_eval import (
            TaskPredictionAccumulator,
            write_holdout_artifacts,
        )

        accumulator = TaskPredictionAccumulator("excision", "bce")
        accumulator.add([1.0, 0.0], [4.0, -4.0], [1.0, 1.0])
        write_holdout_artifacts(tmp_path, {"excision": accumulator}, split="val")

        with (tmp_path / "val_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        metrics = {row["metric"]: float(row["value"]) for row in rows}
        assert metrics["excision_auroc"] == pytest.approx(1.0)
        assert all(row["split"] == "val" for row in rows)


def test_collect_applies_the_same_target_transform_as_the_loss():
    """Otherwise the dump compares raw targets against transformed predictions.

    For the mhcflurry-style affinity encoding — which inverts, so a higher
    score means a stronger binder — that flips the sign of every correlation
    and makes a working model look anti-correlated.
    """
    import torch

    from presto.training.holdout_eval import collect_holdout_predictions

    class _Model:
        def eval(self):
            return self

    class _Spec:
        name = "binding"
        loss_type = "censor"
        pred_paths = (("kd",),)
        # stand-in for normalize_binding_target_log10: monotone decreasing
        target_transform = staticmethod(lambda t: -torch.log10(t))

    class _Batch:
        def to(self, _device):
            return self

    targets = torch.tensor([10.0, 1000.0, 100000.0])
    predictions = -torch.log10(targets)  # a perfect model in transformed space

    accumulators = collect_holdout_predictions(
        model=_Model(),
        loader=[_Batch()],
        device="cpu",
        specs=[_Spec()],
        forward_fn=lambda _m, _b: {"kd": predictions},
        resolve_pred_fn=lambda out, paths: out[paths[0][0]],
        get_target_fn=lambda _b, _s: targets,
        get_mask_fn=lambda _b, _s: torch.ones(3),
    )
    metrics = accumulators["binding"].metrics()
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


class TestProbabilityColumn:
    """`y_pred` is a logit for binary tasks; `y_prob` says what it means.

    The metrics were always right -- `binary_metrics` applies the logistic
    transform before thresholding -- but the dump carried raw logits under a
    column named `y_pred`, so a plausible-looking calibration or Brier
    computation over that file would have been silently wrong. Ranges like
    [-7.38, 3.95] in a column read as probabilities are the tell.
    """

    def _accumulator(self, loss_type, preds):
        from presto.training.holdout_eval import TaskPredictionAccumulator

        acc = TaskPredictionAccumulator("t", loss_type)
        acc.add([1.0] * len(preds), preds, [1.0] * len(preds), [f"s{i}" for i in range(len(preds))])
        return acc

    def test_binary_tasks_carry_a_probability(self):
        rows = self._accumulator("bce", [-7.38, 0.0, 3.95]).rows()
        probs = [r["y_prob"] for r in rows]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert probs[1] == pytest.approx(0.5)

    def test_the_probability_is_the_logistic_of_the_logit(self):
        import math

        rows = self._accumulator("bce", [-2.0, 1.5]).rows()
        for row in rows:
            expected = 1.0 / (1.0 + math.exp(-row["y_pred"]))
            assert row["y_prob"] == pytest.approx(expected)

    def test_regression_tasks_leave_it_empty(self):
        """No logistic transform applies, so inventing one would mislead."""
        rows = self._accumulator("mse", [2.5, -1.0]).rows()
        assert all(r["y_prob"] == "" for r in rows)

    def test_extreme_logits_do_not_overflow(self):
        rows = self._accumulator("bce", [-800.0, 800.0]).rows()
        assert rows[0]["y_prob"] == pytest.approx(0.0)
        assert rows[1]["y_prob"] == pytest.approx(1.0)

    def test_the_column_is_written_to_the_csv(self):
        import csv
        import tempfile
        from pathlib import Path

        from presto.training.holdout_eval import write_holdout_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            write_holdout_artifacts(Path(tmp), {"t": self._accumulator("bce", [0.0, 1.0])})
            with (Path(tmp) / "val_predictions.csv").open() as handle:
                header = next(csv.reader(handle))
        assert "y_prob" in header
