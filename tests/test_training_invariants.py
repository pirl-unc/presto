"""Invariants of the training step itself.

Written after a bug that produced a finite loss, ran to completion, passed
1,417 tests, and learned nothing: the assay panel regressed a normalized-log10
head against a raw-nanomolar target, so one term sat near 142,900 and swamped
every gradient. Validation moved 0.012% across ten epochs.

Nothing in the suite could see it, because every existing test asked "does this
run" or "is this value correct", and the defect was in a *relationship between
magnitudes*. These tests ask the questions that would have caught it:

- does any single term dominate the total?
- do a task's target and its prediction live in the same space?
- does each loss actually respond to its own target?
- does a masked row contribute nothing?
- can the model overfit a handful of examples at all?

The last is the classic sanity check and the strongest: a model that cannot
drive the loss down on eight examples it sees repeatedly is broken regardless
of what any other test says.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from presto.models.presto import Presto  # noqa: E402
from presto.scripts.train_synthetic import (  # noqa: E402
    LOSS_TASK_SPECS,
    compute_loss,
)
from test_gradient_coverage import _every_modality_batch  # noqa: E402


@pytest.fixture
def batch():
    """Function-scoped on purpose.

    Two tests below mutate targets to see whether the loss reacts. A
    `PrestoBatch` holds each target both as an attribute and inside its
    `targets` dict, backed by the same tensor, so restoring the attribute
    leaves the dict copy modified. Sharing one batch across tests made the
    responsiveness test fail only when run after the mask test -- an ordering
    bug in the tests, not in the model. A fresh batch per test costs little and
    removes the class.
    """
    return _every_modality_batch()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Presto(d_model=32, n_layers=2, n_heads=4)


def _terms(model, batch):
    """Loss and its terms, measured deterministically.

    `eval()` is not incidental. The model carries 26 active dropout modules, so
    two identical calls in train mode differ by ~0.004 -- and the tests below
    compare two calls to decide whether a target perturbation moved the loss.
    In train mode they were comparing dropout noise. That passed by luck until
    an unrelated merge shifted the RNG state, which is the whole failure mode
    this file exists to catch, reproduced inside it.
    """
    model.eval()
    loss, parts, _ = compute_loss(model, batch, "cpu")
    values = {
        name: float(value)
        for name, value in parts.items()
        if isinstance(value, (int, float, torch.Tensor)) and float(value) != 0.0
    }
    return float(loss), values


class TestNoTermDominates:
    """The panel bug's signature, stated as a ratio.

    A term orders of magnitude above its neighbours is either mis-scaled or
    mis-spaced, and it takes the gradient budget with it. The ceiling is
    deliberately loose -- uncentered regression targets legitimately start
    high, e.g. log10(kon) is about 5 against a head that starts near 0, giving
    an MSE near 25 at init. 142,900 is not that.
    """

    #: Multiple of the median term that any single term may reach.
    CEILING = 60.0

    def test_no_loss_term_is_orders_of_magnitude_above_the_median(self, model, batch):
        _, values = _terms(model, batch)
        assert len(values) > 10, "loss did not produce a usable set of terms"
        ordered = sorted(values.values())
        median = ordered[len(ordered) // 2]
        assert median > 0, "median loss term is zero; nothing is being supervised"
        offenders = {name: value for name, value in values.items() if value / median > self.CEILING}
        assert offenders == {}, (
            f"these terms exceed {self.CEILING}x the median ({median:.3f}) and "
            f"will dominate the gradient: {offenders}. A term this far out is "
            "usually a target and a prediction in different spaces -- check "
            "whether the head's output is being compared against a raw value."
        )

    def test_the_total_is_finite_and_not_degenerate(self, model, batch):
        total, values = _terms(model, batch)
        assert torch.isfinite(torch.tensor(total)), f"total loss is {total}"
        assert total > 0.0, "total loss is zero; nothing is being supervised"


class TestTargetsAndPredictionsShareASpace:
    """A regression head must predict into the space its target occupies.

    The panel bug was exactly this: a head emitting a normalized log10 offset,
    scored against nanomolar up to 50,000. Comparing ranges catches it without
    knowing anything about the task.
    """

    #: Widest gap allowed between a target and the prediction meant to match
    #: it, in absolute units. Generous: at initialization an untrained head
    #: sits near zero while a target may legitimately be log10(kon) ~ 5.
    MAX_GAP = 50.0

    @staticmethod
    def _dig(mapping, path):
        current = mapping
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _regression_cases(self, model, batch):
        model.eval()
        with torch.no_grad():
            outputs = model(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=getattr(batch, "mhc_class", None),
                flank_n_tok=getattr(batch, "flank_n_tok", None),
                flank_c_tok=getattr(batch, "flank_c_tok", None),
                provenance=getattr(batch, "provenance", None),
            )
        cases = []
        for spec in LOSS_TASK_SPECS:
            if spec.loss_type not in ("mse", "censor", "huber"):
                continue
            attr = getattr(spec, "target_attr", None)
            if not attr:
                continue
            target = getattr(batch, attr, None)
            if target is None:
                continue
            flat = target.reshape(-1).float()
            mask_attr = getattr(spec, "mask_attr", None)
            if mask_attr is not None:
                mask = getattr(batch, mask_attr, None)
                if mask is not None:
                    flat = flat[mask.reshape(-1).float() > 0]
            if flat.numel() == 0:
                continue
            if getattr(spec, "target_transform", None):
                flat = spec.target_transform(flat)
            prediction = None
            for path in spec.pred_paths:
                prediction = self._dig(outputs, path)
                if prediction is not None:
                    break
            if prediction is None:
                continue
            cases.append((spec.name, flat, prediction.reshape(-1).float()))
        return cases

    def test_some_regression_tasks_are_discovered(self, model, batch):
        """Guards the guard against a spec rename."""
        assert len(self._regression_cases(model, batch)) >= 4

    def test_target_and_prediction_are_within_reach(self, model, batch):
        offenders = []
        for name, target, prediction in self._regression_cases(model, batch):
            gap = abs(float(target.mean()) - float(prediction.mean()))
            if gap > self.MAX_GAP:
                offenders.append(
                    f"{name}: target mean {float(target.mean()):.3g} vs "
                    f"prediction mean {float(prediction.mean()):.3g}"
                )
        assert offenders == [], (
            "a head is predicting into a different space than its target "
            "occupies, so the loss cannot be driven down:\n  " + "\n  ".join(offenders)
        )

    def test_targets_are_finite(self, model, batch):
        for name, target, _ in self._regression_cases(model, batch):
            assert torch.isfinite(target).all(), f"{name} has non-finite targets"


class TestMaskedRowsContributeNothing:
    """A masked target must not reach the loss.

    Every task carries a mask because most rows lack most labels. If a masked
    row still contributes, the model is fitting whatever placeholder sits in
    the target tensor -- usually zero -- and the damage scales with how sparse
    the label is, which is to say it is worst exactly where it is hardest to
    notice.
    """

    def test_changing_a_masked_target_does_not_change_the_loss(self, model, batch):
        baseline, _ = _terms(model, batch)
        offenders = []
        for spec in LOSS_TASK_SPECS:
            target_attr = getattr(spec, "target_attr", None)
            mask_attr = getattr(spec, "mask_attr", None)
            if not target_attr or not mask_attr:
                continue
            target = getattr(batch, target_attr, None)
            mask = getattr(batch, mask_attr, None)
            if target is None or mask is None:
                continue
            masked_positions = (mask.reshape(-1) <= 0).nonzero().flatten()
            if masked_positions.numel() == 0:
                continue
            original = target.clone()
            flat = target.reshape(-1)
            flat[masked_positions] = flat[masked_positions] + 1000.0
            try:
                perturbed, _ = _terms(model, batch)
            finally:
                setattr(batch, target_attr, original)
            if abs(perturbed - baseline) > 1e-4:
                offenders.append(
                    f"{spec.name}: loss moved {baseline:.6f} -> {perturbed:.6f} "
                    "when only masked rows changed"
                )
        assert offenders == [], "masked targets are reaching the loss:\n  " + "\n  ".join(offenders)


class TestEachLossRespondsToItsOwnTarget:
    """A task whose loss ignores its target is not being trained.

    This is the complement of the mask test: masked rows must not matter, and
    unmasked ones must. A spec wired to the wrong attribute, or a prediction
    path that silently resolves to None, produces a term that is constant --
    and a constant term looks perfectly healthy in a loss log.
    """

    def test_perturbing_an_unmasked_target_moves_the_loss(self, model, batch):
        baseline, _ = _terms(model, batch)
        checked = 0
        inert = []
        for spec in LOSS_TASK_SPECS:
            if spec.loss_type not in ("mse", "censor", "huber"):
                continue
            target_attr = getattr(spec, "target_attr", None)
            mask_attr = getattr(spec, "mask_attr", None)
            if not target_attr or not mask_attr:
                continue
            target = getattr(batch, target_attr, None)
            mask = getattr(batch, mask_attr, None)
            if target is None or mask is None:
                continue
            live = (mask.reshape(-1) > 0).nonzero().flatten()
            if live.numel() == 0:
                continue
            original = target.clone()
            flat = target.reshape(-1)
            flat[live] = flat[live] + 5.0
            try:
                perturbed, _ = _terms(model, batch)
            finally:
                setattr(batch, target_attr, original)
            checked += 1
            if abs(perturbed - baseline) <= 1e-6:
                inert.append(spec.name)
        assert checked >= 3, f"only {checked} regression tasks had live rows"
        assert inert == [], (
            f"these tasks have supervised rows but their loss does not respond "
            f"to the target: {inert}. The spec is wired to something the loss "
            "does not read."
        )


class TestTheModelCanOverfitATinyBatch:
    """The sanity check that would have caught the panel bug outright.

    A model shown the same eight examples repeatedly must be able to drive the
    loss down on them. If it cannot, something upstream of every metric is
    wrong -- a mis-scaled term, a detached graph, a target the head cannot
    reach. This is the cheapest possible statement of "training works", and the
    suite did not have it while the model was learning nothing.

    Measured against the pre-fix code, the loss moved 0.012% across ten epochs.
    The threshold here is 25% across 60 steps on a fixed batch, which that
    would have failed by three orders of magnitude and any healthy model clears
    comfortably.
    """

    STEPS = 60
    #: Fraction of the initial loss that must be removed.
    REQUIRED_DROP = 0.25

    def test_loss_falls_substantially_on_a_fixed_batch(self, batch):
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        first = None
        best = None
        for _ in range(self.STEPS):
            optimizer.zero_grad()
            loss, _, _ = compute_loss(model, batch, "cpu")
            loss.backward()
            optimizer.step()
            value = float(loss)
            assert torch.isfinite(torch.tensor(value)), "loss became non-finite"
            if first is None:
                first = value
            best = value if best is None else min(best, value)

        drop = (first - best) / first
        assert drop >= self.REQUIRED_DROP, (
            f"loss only fell {drop:.1%} ({first:.4f} -> {best:.4f}) over "
            f"{self.STEPS} steps on a fixed batch. A model that cannot overfit "
            "eight examples is not training; look for a term whose target and "
            "prediction are in different spaces before looking at the "
            "architecture."
        )

    def test_a_dominating_term_would_fail_this(self, batch):
        """The guard's own premise, checked rather than asserted.

        Re-creates the panel bug's shape by scaling one term to the magnitude
        it had, and confirms the overfit check notices. Without this, the
        threshold above is a number nobody has seen fail.
        """
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        first = None
        best = None
        for _ in range(20):
            optimizer.zero_grad()
            loss, _, _ = compute_loss(model, batch, "cpu")
            # A constant of the magnitude the mis-spaced panel term carried.
            # It contributes no gradient, exactly like a term the head cannot
            # move, and it makes the achievable fractional drop negligible.
            polluted = loss + 142_900.0
            polluted.backward()
            optimizer.step()
            value = float(polluted)
            if first is None:
                first = value
            best = value if best is None else min(best, value)

        drop = (first - best) / first
        assert drop < self.REQUIRED_DROP, (
            "a term of the mis-scaled magnitude did not suppress the "
            f"fractional drop ({drop:.1%}); the overfit threshold is not "
            "measuring what it claims to"
        )


class TestEvaluationIsDeterministic:
    """Validation must not be measured through dropout.

    26 `Dropout` modules at p=0.1 are active in this model. If the validation
    pass ran in train mode, every reported val loss would carry that noise, a
    model would be selected on the draw rather than the fit, and the
    early-stopping signal would be partly random -- while looking entirely
    normal in a log.
    """

    def test_eval_mode_gives_the_same_loss_twice(self, batch):
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.eval()
        first, _, _ = compute_loss(model, batch, "cpu")
        second, _, _ = compute_loss(model, batch, "cpu")
        assert float(first) == pytest.approx(float(second), abs=1e-9), (
            "eval-mode loss is not reproducible on an identical batch; "
            "something stochastic is still active"
        )

    def test_train_mode_is_stochastic(self, batch):
        """The other half: if train mode were also deterministic, dropout
        would not be doing anything and the test above would prove nothing."""
        torch.manual_seed(0)
        model = Presto(d_model=32, n_layers=2, n_heads=4)
        model.train()
        torch.manual_seed(1)
        first, _, _ = compute_loss(model, batch, "cpu")
        torch.manual_seed(2)
        second, _, _ = compute_loss(model, batch, "cpu")
        assert float(first) != pytest.approx(float(second), abs=1e-9), (
            "train-mode loss is deterministic, so dropout is inactive and "
            "test_eval_mode_gives_the_same_loss_twice is vacuous"
        )

    def test_the_validation_pass_uses_eval_mode(self):
        """Source-checked: the modes are set in the epoch loop, not reachable
        from a unit test without running a full epoch."""
        import inspect

        from source_probe import occurrences

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic)
        assert occurrences(source, "model.eval()"), "no eval() call in the trainer"
        assert occurrences(source, "model.train()"), "no train() call in the trainer"


class TestCheckpointHoldsTheBestEpoch:
    """Saving the last epoch instead of the best silently ships an overfit
    model.

    This model overfits: on a real 8-epoch run validation bottomed at epoch 5
    (0.6708) and rose to 0.7062 by epoch 8 while training loss kept falling.
    Saving unconditionally would have shipped the epoch-8 weights.
    """

    def test_save_is_nested_inside_the_improvement_guard(self):
        """Structural, not textual.

        A first version searched a 900-character window after the guard for
        `save_model_checkpoint`. That passed even when the save was unindented
        out of the guard and made unconditional, because the string was still
        within the window. Walking the AST asks the question that matters: is
        the call *inside* the `if val_loss < best_val_loss` body.
        """
        import ast
        import inspect

        from presto.scripts import train_synthetic

        source = inspect.getsource(train_synthetic)
        tree = ast.parse(source)

        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "best_val_loss" in ast.dump(node.test)
            and "val_loss" in ast.dump(node.test)
        ]
        assert len(guards) == 1, (
            f"expected exactly one `val_loss < best_val_loss` guard, found "
            f"{len(guards)}; update this test deliberately"
        )

        def _calls(node):
            return {
                child.func.id
                if isinstance(child.func, ast.Name)
                else getattr(child.func, "attr", "")
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            }

        inside = set()
        for statement in guards[0].body:
            inside |= _calls(statement)
        assert "save_model_checkpoint" in inside, (
            "save_model_checkpoint is not inside the `val_loss < best_val_loss` "
            "guard, so every epoch overwrites the checkpoint and the final "
            "(overfit) weights ship instead of the best ones"
        )
