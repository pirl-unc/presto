"""Independent original-field Pandas replay; imports no producer implementation."""

import gzip
import hashlib
import itertools
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

FAMILY = Path(__file__).resolve().parents[1]


def lines(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def original_frame(recipe, condition):
    table = pq.read_table(recipe / "source_tables/filtered_vectors.parquet", filters=[("condition_id", "=", condition)], columns=["fields_json", "source_record_id"])
    expanded = []
    for typed in table.to_pylist():
        r = json.loads(typed["fields_json"])
        for i, (t, v) in enumerate(zip(r["times_pasted"].split(";"), r["int_pasted"].split(";"), strict=True)):
            expanded.append((typed["source_record_id"], i, r["pepSeq"], r["biological_replicate"], int(r["technical_replicate"]), float(t), float(v)))
    df = pd.DataFrame(expanded, columns=["id", "slot", "peptide", "bio", "original_tech", "time", "original"])
    df["il"] = df.peptide.str.replace("I", "L", regex=False)
    df["raw"] = df.groupby(["il", "bio", "original_tech", "time"])["original"].transform("sum")
    df["tech"] = df.original_tech.replace({3: 1, 4: 2})
    zero = df[df.time == 0].groupby(["peptide", "bio", "tech"]).raw.max()
    noise = float(zero.median())
    df = df.join(zero.clip(lower=noise).rename("baseline"), on=["peptide", "bio", "tech"])
    df["residual"] = df.raw - df.baseline
    df["keep"] = (df.time > 0) & (df.residual > 0)
    full = df.groupby(["peptide", "bio", "time"]).raw.mean().rename("unthresholded")
    retained = df[df.keep].groupby(["peptide", "bio", "time"])[["raw", "residual", "baseline"]].mean()
    means = retained.join(full).to_dict("index")
    return df, means, noise


def main():
    started = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    recipe = Path(cfg["recipe_directory"])
    for path, expected in cfg["input_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    selected = {r["curve_id"]: r for r in json.loads((recipe / "examples.json").read_text()) if r["split"] == "train"}
    verified_curves = set(); point_values = {}; checked = 0; contributors = 0; maxrelative = 0.0; mutations = {}
    for condition, group in itertools.groupby(lines(FAMILY / "results/curve-layers.jsonl.gz"), key=lambda r: r["condition_id"]):
        df, means, noise = original_frame(recipe, condition)
        lookup = df.set_index(["id", "slot"]).to_dict("index")
        for curve in group:
            e = selected[curve["curve_id"]]; verified_curves.add(e["curve_id"])
            near(noise, curve["global_noise"])
            assert curve["split"] == "train" and curve["common_positive_times"] == e["common_positive_times"]
            expected = {r["time_hours"]: r["target_value"] for r in e["observations"]}
            for point in curve["points"]:
                near(point["corrected"], expected[point["time"]])
                for alias in e["peptide_aliases"]:
                    m = means[alias, e["biological_replicate_raw"], point["time"]]
                    for field, name in (("corrected", "residual"), ("restored", "raw"), ("unthresholded", "unthresholded"), ("background", "baseline")):
                        near(point[field], m[name]); maxrelative = max(maxrelative, abs(point[field] / m[name] - 1))
                for source in point["contributors"]:
                    raw = lookup[source["vector_id"], source["slot"]]
                    near(raw["original"], source["original_intensity"])
                    near(raw["raw"], source["intensity"])
                    near(raw["baseline"], source["baseline"])
                    assert bool(raw["keep"]) == source["retained"]
                    assert raw["original_tech"] == source["technical_original"] and raw["tech"] == source["technical_remapped"]
                    contributors += 1
                candidate = None if len(mutations)==2 else df[(df.peptide == e["peptide_aliases"][0]) & (df.bio == e["biological_replicate_raw"]) & (df.time == point["time"]) & df.keep]
                if "baseline" not in mutations:
                    delta = float(candidate.residual.min()) * 0.01
                    altered = float((candidate.raw - candidate.baseline - delta).mean())
                    try: near(point["corrected"], altered)
                    except AssertionError: mutations["baseline"] = dict(curve_id=e["curve_id"],time=point["time"],original=point["corrected"],changed=altered,baseline_increment=delta)
                    else: raise AssertionError("Failed to detect baseline perturbation")
                if "contributor_mean" not in mutations and len(candidate)>1 and candidate.residual.max()-candidate.residual.min()>1e-5:
                    altered = float(candidate.residual.iloc[1:].mean())
                    try: near(point["corrected"], altered)
                    except AssertionError: mutations["contributor_mean"] = dict(curve_id=e["curve_id"],time=point["time"],original=point["corrected"],changed=altered,removed_contributor=[candidate.iloc[0]["id"],int(candidate.iloc[0]["slot"])])
                    else: raise AssertionError("Failed to detect contributor removal")
                point_values[e["curve_id"], point["time"]] = {k: point[k] for k in ("corrected", "restored", "unthresholded", "background")}
                checked += 1
    assert verified_curves == selected.keys()
    ratios = pd.DataFrame(lines(FAMILY / "results/ratios.jsonl.gz"))
    expected_n = sum(len(e["common_positive_times"]) - 1 for e in selected.values())
    assert len(ratios) == expected_n * 3
    conditions = {r["condition_id"]: r["construct_id"] for r in selected.values()}
    parents = set(conditions.values())
    ncurves = pd.Series([e["condition_id"] for e in selected.values()]).value_counts().to_dict()
    nconditions = pd.Series(list(conditions.values())).value_counts().to_dict()
    mates = {}
    for e in selected.values():
        mates.setdefault((e["condition_id"], e["peptide_IL_sequence"]), []).append(e["curve_id"])
    for r in ratios.itertuples():
        e = selected[r.curve_id]; ref = e["common_positive_times"][0]
        a = point_values[r.curve_id, ref]; b = point_values[r.curve_id, r.time]
        near(r.target, math.log10(b[r.layer]) - math.log10(a[r.layer]))
        mate = next(x for x in mates[e["condition_id"], e["peptide_IL_sequence"]] if x != r.curve_id)
        near(r.other_label_prediction, math.log10(point_values[mate, r.time][r.layer] / point_values[mate, ref][r.layer]))
        near(r.weight, 1 / (len(parents) * nconditions[e["construct_id"]] * ncurves[e["condition_id"]] * (len(e["common_positive_times"]) - 1)))
        near(r.reference_subtracted_fraction, a["background"] / a["restored"])
        boundaries = (e["start_0based"] > 0) + (e["end_0based"] < len(e["parent_sequence"]))
        near(r.excess, r.target - boundaries * math.log10(r.time / ref))
    metric_error = 0.0
    for stored in json.loads((FAMILY / "results/metrics.json").read_text()):
        sub = ratios[ratios.layer == stored["layer"]]
        if stored["parent"] != "all": sub = sub[sub.construct_id == stored["parent"]]
        y = sub.target.to_numpy(); p = sub[stored["predictor"]].to_numpy(); w = sub.weight.to_numpy(); w /= w.sum()
        expected = dict(n=len(sub), macro_rmse=np.sqrt(np.dot(w, (y-p)**2)), pooled_rmse=np.sqrt(np.mean((y-p)**2)),
                        prediction_sd=np.std(p), target_sd=np.std(y), pearson=None if np.ptp(p)==0 else pearsonr(y,p).statistic,
                        spearman=None if np.ptp(p)==0 else spearmanr(y,p).statistic)
        for key,value in expected.items():
            if value is None: assert stored[key] is None
            else: near(value,stored[key]); metric_error=max(metric_error,abs(value-stored[key]))
    triplets = list(lines(FAMILY / "results/triplets.jsonl.gz"))
    for r in triplets:
        a,b,c = r["times"]; values=[math.log10(point_values[r["curve_id"],t][r["layer"]]) for t in (a,b,c)]
        near(r["chord_minus_middle"], ((c-b)*values[0]+(b-a)*values[2])/(c-a)-values[1])
    assert set(mutations) == {"baseline", "contributor_mean"}
    result = dict(passed=True,no_producer_imports=True,curves=len(verified_curves),positive_source_points=checked,
                  individual_contributors_checked=contributors,ratios_verified=len(ratios),triplets_verified=len(triplets),
                  maximum_layer_relative_error=maxrelative,maximum_metric_error=metric_error,mutation_checks=mutations,
                  wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (FAMILY / "results/independent-verification.json").write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == "__main__": main()
