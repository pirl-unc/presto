"""Twenty fixed development fits and four explicitly descriptive full fits."""

import csv
import json

from common import ROOT, read, write
from model import ARMS, fit, fit_fold, predict, summarize


def main():
    rows = read(ROOT / "results/admitted.json")
    levels = sorted({r["nominal_level_id"] for r in rows})
    assert len(levels) == 5
    models, predictions, full_models, full_predictions = [], [], [], []
    for arm in ARMS:
        for level in levels:
            model, held, values = fit_fold(rows, arm, level)
            models.append(model)
            predictions.extend(
                {
                    **r,
                    "arm": arm,
                    "held_level": level,
                    "prediction": value,
                    "train_ids_sha256": model["train_ids_sha256"],
                }
                for r, value in zip(held, values, strict=True)
            )
        model = fit(rows, arm)
        full_models.append(model)
        full_predictions.extend(
            {**r, "arm": arm, "prediction": value}
            for r, value in zip(rows, predict(model, rows), strict=True)
        )
    write(ROOT / "results/fold-models.json", models)
    write(ROOT / "results/full-models.json", full_models)
    write(ROOT / "results/development-predictions.json", predictions)
    write(ROOT / "results/descriptive-full-predictions.json", full_predictions)
    summary = summarize(predictions, models)
    summary.update(
        n_fold_fits=len(models),
        n_full_fits=len(full_models),
        evaluation="selected-panel leave-level-out development; no independent test",
        full_goal_achieved=False,
    )
    write(ROOT / "results/summary.json", summary)
    with (ROOT / "results/comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", *summary["by_arm"][ARMS[0]]["overall"]])
        writer.writeheader()
        for arm in ARMS:
            writer.writerow({"arm": arm, **summary["by_arm"][arm]["overall"]})
    print(
        json.dumps(
            {
                "metrics": {a: summary["by_arm"][a]["overall"] for a in ARMS},
                "gates": summary["gates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
