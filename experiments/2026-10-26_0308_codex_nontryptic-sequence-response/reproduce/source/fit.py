"""Fixed nested precursor-grouped response comparison with every prediction saved."""

import time

from common import ROOT, read, request, write
from model import FEATURES, fit, metrics, predict, select_alpha


def main():
    req = request()
    cfg = read(ROOT / "reproduce/launch.json")
    rows = read(req["admitted"])
    folds = read(req["folds"])
    out = ROOT / "results" / req["output_subdir"]
    out.mkdir(exist_ok=False)
    models, inner_predictions, outer_predictions, selections = [], [], [], []
    started = time.monotonic()
    for fold in folds:
        held = fold["held_gene"]
        train = [r for r in rows if r["gene"] != held]
        test = [r for r in rows if r["gene"] == held]
        assert {r["sequence"] for r in test} == set(fold["held_sequences"])
        assert not {r["sequence"] for r in train} & {r["sequence"] for r in test}
        chosen, losses, inner_models, inner = select_alpha(train, cfg["alphas"])
        selections.append({"outer_gene": held, "chosen_alpha": chosen, "inner_losses": losses})
        offset = len(models)
        for m in inner_models:
            m.update(model_id=len(models), outer_gene=held, level="inner")
            models.append(m)
        for p in inner:
            p.update(model_id=offset + p.pop("local_model_id"), outer_gene=held)
            inner_predictions.append(p)
        for arm in ("unit", "slope", "sequence"):
            m = fit(train, arm, chosen if arm == "sequence" else 0)
            m.update(model_id=len(models), outer_gene=held, level="outer")
            models.append(m)
            for row, value in zip(test, predict(m, test), strict=True):
                outer_predictions.append(
                    {
                        **row,
                        "arm": arm,
                        "model_id": m["model_id"],
                        "outer_gene": held,
                        "prediction": float(value),
                        "training_ids_sha256": m["training_ids_sha256"],
                    }
                )
    summary = {
        "features": FEATURES,
        "n_fits": len(models),
        "fit_elapsed_seconds": time.monotonic() - started,
        "evaluation": "nested_precursor_grouped_development_not_independent_biological_test",
        "arms": {},
    }
    for arm in ("unit", "slope", "sequence"):
        records = [r for r in outer_predictions if r["arm"] == arm]
        assert len(records) == len(rows)
        summary["arms"][arm] = metrics(records, [r["prediction"] for r in records])
        for group_field in ("sequence", "dose_fmol", "gene"):
            summary["arms"][arm]["per_" + group_field] = {
                str(value): metrics(group, [r["prediction"] for r in group])
                for value in sorted({r[group_field] for r in records})
                if (group := [r for r in records if r[group_field] == value])
            }
    baseline, sequence = (summary["arms"][a] for a in ("slope", "sequence"))
    improvement = 1 - sequence["equal_peptide_rmse"] / baseline["equal_peptide_rmse"]
    gates = {
        "rmse_reduction_at_least_10_percent": improvement >= 0.1,
        "all_outer_learned_dose_slopes_positive": all(
            m["dose_slope"] > 0 for m in models if m["level"] == "outer"
        ),
        "dose_adjusted_sd_at_least_25_percent": sequence["dose_adjusted_prediction_sd"]
        >= 0.25 * sequence["dose_adjusted_target_sd"],
    }
    summary.update(
        relative_rmse_improvement=float(improvement),
        gates=gates,
        passed=all(gates.values()),
        biological_model_promoted=False,
        n_inner_predictions=len(inner_predictions),
        n_outer_predictions=len(outer_predictions),
    )
    for name, value in [
        ("models", models),
        ("inner-predictions", inner_predictions),
        ("outer-predictions", outer_predictions),
        ("selections", selections),
        ("summary", summary),
    ]:
        write(out / (name + ".json"), value)
    print(
        {"n_fits": len(models), "relative_rmse_improvement": float(improvement), "gates": gates},
        flush=True,
    )


if __name__ == "__main__":
    main()
