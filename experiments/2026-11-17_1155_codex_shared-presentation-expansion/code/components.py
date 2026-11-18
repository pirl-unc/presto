"""Descriptive monoallelic component readouts from existing validation exports."""

import gzip
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

E = Path(__file__).resolve().parents[1]


def main():
    rows = []
    paths = [
        (p, p.parents[1].name, None)
        for p in sorted((E / "results/fits").glob("*-n6000/raw/validation-predictions.jsonl.gz"))
    ]
    for anchor in ["parent", "previous-default"]:
        paths.extend(
            (p, anchor, p.name.split("-")[0])
            for p in sorted((E / "results/evaluation" / anchor).glob("*-predictions.jsonl.gz"))
        )
    paths.extend(
        (p, p.parent.name, "matched")
        for p in sorted((E / "results/evaluation").glob("*/matched-predictions.jsonl.gz"))
        if p.parent.name not in ("parent", "previous-default")
    )
    paths.extend(
        (p, p.parent.name, None)
        for p in sorted(
            (E / "results/evaluation").glob("*-terminal/validation-predictions.jsonl.gz")
        )
    )
    for path, model, population in paths:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                r = json.loads(line)
                if r["endpoint"] != "elution_observation":
                    continue
                candidates = r["diagnostics"]["candidates"]
                if len(candidates) != 1 or len(candidates[0]["registers"]) != 1:
                    continue
                register = candidates[0]["registers"][0]
                terms = register["elution_terms"]
                parts = {
                    k: float(terms.get(k, 0))
                    for k in ["binding", "processing", "selection", "stability"]
                }
                presentation = sum(parts.values())
                assert abs(presentation - r["component_scores"]["elution_presentation"]) < 1e-5
                provenance = r["observation"]["provenance"]
                pop = population or (
                    "expanded"
                    if provenance.get("source_family")
                    else "apc"
                    if r["source"] == "hitlist_native_ms"
                    else "broad"
                )
                rows.append(
                    {
                        "model": model,
                        "population": pop,
                        "class": register["class"],
                        "id": r["observation_id"],
                        "target": r["target"],
                        "registers": len(register["starts"]),
                        "register_entropy": register["entropy"],
                        "presentation": presentation,
                        "final": r["prediction"],
                        **parts,
                    }
                )
    data = pd.DataFrame(rows)
    data.to_parquet(E / "results/analysis/monoallelic-components.parquet", index=False)
    reference = {}
    result = []
    for (model, pop, cls), group in data.groupby(["model", "population", "class"]):
        key = (pop, cls)
        ids = set(group.id)
        assert len(ids) == len(group)
        if key in reference:
            assert ids == reference[key], (model, key, "different monoallelic cohort")
        else:
            reference[key] = ids
        if group.target.nunique() != 2:
            continue
        result.append(
            {
                "model": model,
                "population": pop,
                "class": cls,
                "rows": len(group),
                **{
                    name + "_auroc": roc_auc_score(group.target, group[name])
                    for name in ["binding", "processing", "selection", "presentation", "final"]
                },
            }
        )
    metrics = pd.DataFrame(result)
    metrics.to_csv(E / "results/analysis/monoallelic-component-metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
