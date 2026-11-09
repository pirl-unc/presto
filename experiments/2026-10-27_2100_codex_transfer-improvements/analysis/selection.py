"""Seal a decision from independently recomputed validation metrics only."""

from pathlib import Path
import json
import time
import numpy as np
from verify import FAMILY, CONDITIONS, load, metrics, assert_metric, sha

PRIOR = FAMILY.parent / '2026-10-27_1930_codex_modal-cross-class'


def gate(candidate, reference):
    # Inputs have axes seed, class, metric(rho, RMSE).
    delta = candidate[:, :, 0] - reference[:, :, 0]
    class_delta = delta.mean(0)
    rmse_ratio = candidate[:, :, 1].mean(0) / reference[:, :, 1].mean(0)
    checks = dict(
        macro_gain_at_least_02=bool(delta.mean() >= 0.02),
        both_seeds_improve=bool((delta.mean(1) > 0).all()),
        no_class_rho_drop_over_01=bool((class_delta >= -0.01).all()),
        no_class_rmse_increase_over_2pct=bool((rmse_ratio <= 1.02).all()),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        macro_delta=float(delta.mean()),
        class_rho_delta=class_delta.tolist(),
        seed_macro_delta=delta.mean(1).tolist(),
        class_rmse_ratio=rmse_ratio.tolist(),
    )


def validation(family, name, classes):
    directory = family / 'results' / name
    report = json.loads((directory / 'report.json').read_text())
    resolved = json.loads((directory / 'resolved-config.json').read_text())
    scale = resolved['scales']['ic50']['scale']
    records = load(directory / 'validation-predictions.jsonl.gz')
    result = {}
    for kind in classes:
        actual = metrics([r for r in records if r['mhc_classes'] == [kind]], scale)
        assert_metric(actual, report['summary']['metrics']['validation']['ic50']['by_class'][kind])
        result[kind] = actual
    return result, sha(directory / 'validation-predictions.jsonl.gz')


def main():
    target = FAMILY / 'results/validation-decision.json'
    if target.exists():
        raise ValueError('decision already sealed')
    current = {}
    hashes = {}
    for name, spec in CONDITIONS.items():
        current[name], hashes[name] = validation(FAMILY, name, spec['classes'])
    historical = {}
    for seed in (42, 43):
        for variant, classes in (
            ('shared-joint', ['I', 'II']),
            ('shared-I', ['I']),
            ('shared-II', ['II']),
        ):
            name = f'{variant}-s{seed}'
            historical[name], hashes['historical:' + name] = validation(PRIOR, name, classes)

    def matrix(variant, old=False):
        source = historical if old else current
        return np.array(
            [
                [
                    [source[f'{variant}-s{seed}'][kind][m] for m in ('spearman', 'rmse')]
                    for kind in ('I', 'II')
                ]
                for seed in (42, 43)
            ]
        )

    pairs = [
        ('pooled-balanced-joint', 'pooled-natural-joint'),
        ('interaction-natural-joint', 'pooled-natural-joint'),
        ('interaction-balanced-joint', 'pooled-balanced-joint'),
        ('interaction-balanced-joint', 'interaction-natural-joint'),
    ]
    gates = {f'{a} vs {b}': gate(matrix(a), matrix(b)) for a, b in pairs}
    gates['2000 vs historical 1000'] = gate(
        matrix('pooled-natural-joint'), matrix('shared-joint', True)
    )
    own_deltas = {}
    for kind in ('I', 'II'):
        own_deltas[kind] = float(
            np.mean(
                [
                    current[f'interaction-{kind}-s{s}'][kind]['spearman']
                    - historical[f'shared-{kind}-s{s}'][kind]['spearman']
                    for s in (42, 43)
                ]
            )
        )
    own_guard = all(value >= -0.02 for value in own_deltas.values())
    # Apply only passing matched changes, allowing the sampling/architecture
    # interaction to differ. Choose the highest validation macro among reachable
    # candidates; no new test metric enters eligibility or ordering.
    eligible = {"pooled-natural-joint"}
    if gates["pooled-balanced-joint vs pooled-natural-joint"]["passed"]:
        eligible.add("pooled-balanced-joint")
    if own_guard:
        if gates["interaction-natural-joint vs pooled-natural-joint"]["passed"]:
            eligible.add("interaction-natural-joint")
        if (
            "pooled-balanced-joint" in eligible
            and gates["interaction-balanced-joint vs pooled-balanced-joint"]["passed"]
        ) or (
            "interaction-natural-joint" in eligible
            and gates["interaction-balanced-joint vs interaction-natural-joint"]["passed"]
        ):
            eligible.add("interaction-balanced-joint")
    preferred = max(sorted(eligible), key=lambda name: matrix(name)[:, :, 0].mean())
    interaction = preferred.startswith("interaction-")
    balance = "balanced" in preferred
    result = dict(
        status='sealed_before_new_test_analysis',
        created_unix=time.time(),
        analysis_sha256=sha(Path(__file__)),
        prediction_hashes=hashes,
        validation_metrics=current,
        historical_validation_metrics=historical,
        gates=gates,
        source_only_within_class_rho_delta=own_deltas,
        source_only_guard=own_guard,
        preferred_variant=preferred,
        eligible_variants=sorted(eligible),
        promote_interaction=interaction,
        promote_balancing=balance,
        test_used_for_decision=False,
    )
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in ('validation_metrics', 'historical_validation_metrics', 'prediction_hashes')
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
