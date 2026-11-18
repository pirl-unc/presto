"""Compare fixed-checkpoint APC input controls without claiming causal effects."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from analyze import bootstrap, complete_pairs, records
from apc_controls import canonical, signature
from report import table

E = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    standard = pd.read_parquet(E / "results/analysis/predictions.parquet")
    for run in args.runs.split(","):
        folder = E / "results/context-ablation" / run
        receipt = json.loads((folder / "receipt.json").read_text())
        assert not receipt["test_accessed"]
        out = folder / "analysis"
        out.mkdir(exist_ok=True)
        collected, input_signatures = [], {}
        for pop in ("apc", "expanded", "matched"):
            for mode in ("original", "unknown", "within-study-shuffle"):
                path = folder / f"{pop}-{mode}-predictions.jsonl.gz"
                collected.extend(records(path, mode, pop))
                with gzip.open(path, "rt") as stream:
                    input_signatures[(pop, mode)] = {
                        r["observation_id"]: canonical(signature(r["observation"]["context"]))
                        for r in map(json.loads, stream)
                    }
        data = pd.DataFrame(collected)
        data.to_parquet(out / "predictions.parquet", index=False)
        metrics = []
        for (pop, cls), group in data.groupby(["population", "class"]):
            original = group[group.model == "original"].set_index("id")
            baseline = standard[
                (standard.model == run) & (standard.population == pop) & (standard["class"] == cls)
            ].set_index("id")
            assert set(original.index) == set(baseline.index)
            baseline = baseline.loc[original.index]
            np.testing.assert_allclose(original.prediction, baseline.prediction, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(original.target, baseline.target)
            for mode, rows in group.groupby("model"):
                assert complete_pairs(rows)
                current = rows.set_index("id").loc[original.index]
                np.testing.assert_array_equal(original.target, current.target)
                np.testing.assert_array_equal(original.pair, current.pair)
                paired = current.pivot(index="pair", columns="target", values="prediction")
                pairwin = ((paired[1] > paired[0]) + 0.5 * (paired[1] == paired[0])).mean()
                delta = abs(current.prediction - original.prediction)
                if receipt["negative_control"]:
                    assert delta.max() < 1e-6
                changed = sum(
                    input_signatures[(pop, mode)][i] != input_signatures[(pop, "original")][i]
                    for i in original.index
                )
                metrics.append(
                    {
                        "population": pop,
                        "class": cls,
                        "mode": mode,
                        "rows": len(rows),
                        "changed_apc_input_rows": changed,
                        "auroc": roc_auc_score(rows.target, rows.prediction),
                        "auprc": average_precision_score(rows.target, rows.prediction),
                        "bce": rows.loss.mean(),
                        "pair_win": pairwin,
                        "max_logit_change": delta.max(),
                    }
                )
        m = pd.DataFrame(metrics)
        m.to_csv(out / "metrics.csv", index=False)
        for cluster in ("protein", "condition"):
            bootstrap(data, "original", out, cluster)
        text = [
            f"# APC input controls: {run}",
            "",
            table(m),
            "",
            "Original scores reproduce the collected predictions for this same checkpoint. "
            "Both pair members retain the same APC signature; source flanks, MHC candidates, "
            "outcomes and other inputs are preserved. Unknown means missing state, not untreated. "
            "Whole signatures are shuffled within study/class; groups with one signature cannot "
            "identify a state effect. This is an inference sensitivity/association test, "
            "not a trained-shuffle control or isolated causal intervention estimate.",
            "",
        ]
        (out / "report.md").write_text("\n".join(text))
        print(run, "verified", len(data), "predictions")


if __name__ == "__main__":
    main()
