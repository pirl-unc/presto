"""Independent context, augmented-SVD fit and full development-result replay."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PKG = Path(CFG["package"])
MAN = json.loads((PKG / "manifest.json").read_text())
AA = "ACDEFGHIKLMNPQRSTVWY"
TOKENS = [*AA, "OTHER", "BOUNDARY"]
numeric_checks, maximum_error = 0, 0.0


def check(a, b, path=""):
    global numeric_checks, maximum_error
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            check(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            check(x, y, path + f"/{i}")
    elif isinstance(a, (float, np.floating)):
        assert math.isfinite(a) and math.isfinite(b), path
        maximum_error = max(maximum_error, abs(a - b))
        assert math.isclose(
            a, b, rel_tol=CFG["metric_rtol"], abs_tol=CFG["metric_atol"]
        ), (path, a, b)
        numeric_checks += 1
    else:
        assert a == b, (path, a, b)


def read(relative, filters):
    p = PKG / relative
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest() == MAN["files"][relative]["sha256"]
    )
    return pq.read_table(p, filters=filters).to_pylist()


def partition(group, namespace, n):
    return (
        int.from_bytes(
            hashlib.sha256((namespace + "|" + group).encode()).digest(), "big"
        )
        % n
    )


def outer(group):
    return partition(group, CFG["outer_namespace"], 5)


def inner(group):
    return partition(group, CFG["inner_namespace"], 3)


def endpoint(parent, peptide):
    indices = [
        i
        for i in range(len(parent) - len(peptide) + 1)
        if parent.startswith(peptide, i)
    ]
    assert indices
    counts = defaultdict(int)
    for aa in peptide:
        counts[aa] += 1
    raw = [
        math.log(len(peptide)),
        *[counts[a] / len(peptide) for a in AA],
        *[float(a == peptide[0]) for a in AA],
        *[float(a == peptide[-1]) for a in AA],
    ]
    boundary = [[0.0] * 22 for _ in range(4)]
    spans = []
    for i in indices:
        positions = [i - 2, i - 1, i + len(peptide), i + len(peptide) + 1]
        tokens = []
        for slot, j in enumerate(positions):
            token = (
                "BOUNDARY"
                if j < 0 or j >= len(parent)
                else parent[j]
                if parent[j] in AA
                else "OTHER"
            )
            tokens.append(token)
            boundary[slot][TOKENS.index(token)] += 1 / len(indices)
        spans.append(
            dict(start_1based=i + 1, end_1based=i + len(peptide), tokens=tokens)
        )
    return np.array(raw + [v for block in boundary for v in block]), spans


rows = sorted(
    read("typed/examples.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(rows) == 158 and all(r["split"] == "train" for r in rows)
parents = {
    p["parent_sequence_sha256"]: p
    for p in read("typed/parents.parquet", [("split", "=", "train")])
}
assert len(parents) == 56
pids = sorted({r[key] for r in rows for key in ("peptide_id", "reference_peptide_id")})
origins = read(
    "source_tables/origins.parquet",
    [("peptide_id", "in", pids), ("reference_sequence_sha256", "in", sorted(parents))],
)
mapped = defaultdict(list)
for record in origins:
    mapped[record["peptide_id"], record["reference_sequence_sha256"]].append(record)
expected_context = {}
xs = {k: [] for k in ("length", "sequence", "boundary")}
for row in rows:
    digest = row["parent_sequence_sha256"]
    parent = parents[digest]["parent_sequence"]
    assert hashlib.sha256(parent.encode()).hexdigest() == digest
    end_vectors = []
    for prefix in ("peptide", "reference_peptide"):
        pid, peptide = row[prefix + "_id"], row[prefix + "_sequence"]
        vector, spans = endpoint(parent, peptide)
        source = mapped[pid, digest]
        assert {(s["start_1based"], s["end_1based"]) for s in spans} == {
            (o["candidate_start_1based"], o["candidate_end_1based"]) for o in source
        }
        expected_context[pid + "|" + digest] = dict(
            peptide_id=pid,
            peptide_sequence=peptide,
            parent_sequence_sha256=digest,
            occurrences=spans,
            source_origin_ids=sorted(o["origin_id"] for o in source),
            endpoint_features=vector.tolist(),
        )
        end_vectors.append(vector)
    difference = end_vectors[0] - end_vectors[1]
    xs["length"].append(difference[:1])
    xs["sequence"].append(difference[:61])
    xs["boundary"].append(difference)
check(
    expected_context, json.loads((ROOT / "results/endpoint-contexts.json").read_text())
)
xs = {k: np.array(v) for k, v in xs.items()}
with np.load(ROOT / "results/features.npz", allow_pickle=False) as stored:
    for kind in xs:
        check(xs[kind].tolist(), stored[kind].tolist(), kind)
ys = np.array([r["target_log10_ratio_change"] for r in rows])
ws = np.array([r["weight"] for r in rows])
index_by_id = {r["example_id"]: i for i, r in enumerate(rows)}
group_ids = [r["dependency_group_id"] for r in rows]
groups = sorted(set(group_ids))
assert len(groups) == 55
prepare = json.loads((ROOT / "results/preparation.json").read_text())
check([r["example_id"] for r in rows], prepare["example_ids"])
for fold in range(5):
    omitted = [g for g in groups if outer(g) == fold]
    fitting = [g for g in groups if outer(g) != fold]
    counts = {str(j): sum(inner(g) == j for g in fitting) for j in range(3)}
    assert len(omitted) >= 5 and min(counts.values()) >= 5
    check(
        dict(
            development_components=omitted,
            fitting_components=fitting,
            inner_counts=counts,
        ),
        prepare["folds"][str(fold)],
    )


def svd_fit(indices, kind, alpha, original):
    x, y, w = xs[kind][indices], ys[indices], ws[indices]
    w = w / w.sum()
    scales = np.sqrt(np.einsum("i,ij,ij->j", w, x, x))
    active = np.where(scales != 0)[0]
    z = x[:, active] / scales[active]
    augmented = np.vstack(
        (np.sqrt(w)[:, None] * z, math.sqrt(alpha) * np.eye(len(active)))
    )
    target = np.r_[np.sqrt(w) * y, np.zeros(len(active))]
    beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
    coef = np.zeros(x.shape[1])
    coef[active] = beta
    mse = float(np.dot(w, (z @ beta - y) ** 2))
    penalty = float(alpha * np.sum(beta**2))
    check(
        dict(
            representation=kind,
            alpha=float(alpha),
            scales=scales.tolist(),
            active_coordinates=active.tolist(),
            coefficients=coef.tolist(),
            n_raw_features=x.shape[1],
            n_solved_coordinates=len(active),
            design_rank=int(np.linalg.matrix_rank(z)) if len(active) else 0,
            n_nonzero_coefficients=int(np.count_nonzero(beta)),
            stationarity_max_abs=original["stationarity_max_abs"],
            training_mse=mse,
            penalty=penalty,
            training_objective=mse + penalty,
            training_example_ids=[rows[i]["example_id"] for i in indices],
            training_components=sorted({group_ids[i] for i in indices}),
        ),
        original,
    )
    original_beta = np.array(original["coefficients"])[active]
    gradient = z.T @ (w * (z @ original_beta - y)) + alpha * original_beta
    assert np.max(np.abs(gradient), initial=0) < 1e-10
    assert original["stationarity_max_abs"] < 1e-10
    return active, scales, beta


def predict(indices, kind, fitted):
    active, scales, beta = fitted
    return (xs[kind][indices][:, active] / scales[active]) @ beta


inner_records = json.loads((ROOT / "results/inner-fits.json").read_text())
inner_predictions = pq.read_table(
    ROOT / "results/inner-predictions.parquet"
).to_pylist()
lookup = {
    (r["outer_fold"], r["kind"], r["alpha"], r["fold"], r["example_id"]): r
    for r in inner_predictions
}
assert len(lookup) == len(inner_predictions)
assert len(inner_records) == 225
losses = defaultdict(list)
for record in inner_records:
    fold, kind, alpha, inner_fold = (
        record["outer_fold"],
        record["kind"],
        record["alpha"],
        record["fold"],
    )
    train = [
        i
        for i, g in enumerate(group_ids)
        if outer(g) != fold and inner(g) != inner_fold
    ]
    valid = [
        i
        for i, g in enumerate(group_ids)
        if outer(g) != fold and inner(g) == inner_fold
    ]
    fitted = svd_fit(train, kind, alpha, record["fitted"])
    values = predict(valid, kind, fitted)
    weights = ws[valid] / ws[valid].sum()
    loss = float(weights @ (values - ys[valid]) ** 2)
    check(loss, record["loss"])
    losses[fold, kind, alpha].append(loss)
    for i, value, weight in zip(valid, values, weights):
        check(
            dict(
                outer_fold=fold,
                example_id=rows[i]["example_id"],
                kind=kind,
                alpha=alpha,
                fold=inner_fold,
                target=float(ys[i]),
                prediction=float(value),
                weight=float(weight),
            ),
            lookup.pop((fold, kind, alpha, inner_fold, rows[i]["example_id"])),
        )
assert not lookup
models = json.loads((ROOT / "results/models.json").read_text())
oof = pq.read_table(ROOT / "results/development-oof-predictions.parquet").to_pylist()
lookup = {(r["example_id"], r["kind"]): r for r in oof}
assert len(models) == 20 and len(lookup) == len(oof) == 632
predictions = {k: np.zeros(len(rows)) for k in CFG["representations"]}
for record in models:
    fold, kind = record["outer_fold"], record["kind"]
    train = [i for i, g in enumerate(group_ids) if outer(g) != fold]
    valid = [i for i, g in enumerate(group_ids) if outer(g) == fold]
    if kind == "zero":
        values = np.zeros(len(valid))
        check(record["fitted"], dict(representation="zero"))
    else:
        means = {
            alpha: float(np.mean(losses[fold, kind, alpha]))
            for alpha in CFG["alpha_grid"]
        }
        selected = min(CFG["alpha_grid"], key=lambda a: (means[a], -a))
        assert selected == record["fitted"]["alpha"]
        check({str(k): v for k, v in means.items()}, record["selection_mean_mse"])
        values = predict(valid, kind, svd_fit(train, kind, selected, record["fitted"]))
    predictions[kind][valid] = values
    for i, value in zip(valid, values):
        check(
            dict(
                example_id=rows[i]["example_id"],
                dependency_group_id=group_ids[i],
                parent_sequence_sha256=rows[i]["parent_sequence_sha256"],
                original_split="train",
                outer_fold=fold,
                kind=kind,
                target=float(ys[i]),
                prediction=float(value),
                weight=float(ws[i]),
            ),
            lookup.pop((rows[i]["example_id"], kind)),
        )
assert not lookup


def ranks(a):
    order = np.argsort(a, kind="stable")
    result = np.empty(len(a), float)
    start = 0
    while start < len(a):
        end = start + 1
        while end < len(a) and a[order[end]] == a[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def corr(a, b, weights):
    if min(a) == max(a) or min(b) == max(b):
        return None
    ac, bc = a - sum(a * weights), b - sum(b * weights)
    return float(
        sum(weights * ac * bc) / math.sqrt(sum(weights * ac**2) * sum(weights * bc**2))
    )


def metric(indices, values):
    y, p, weights = ys[indices], values[indices], ws[indices]
    weights = weights / weights.sum()
    uniform = np.ones(len(y)) / len(y)
    mse = math.fsum(weights * (p - y) ** 2)
    target_sd = math.sqrt(math.fsum(weights * (y - math.fsum(weights * y)) ** 2))
    prediction_sd = (
        0.0
        if min(p) == max(p)
        else math.sqrt(math.fsum(weights * (p - math.fsum(weights * p)) ** 2))
    )
    nz = y != 0
    return dict(
        n_contrasts=len(y),
        n_components=len({group_ids[i] for i in indices}),
        loss=mse,
        mse=mse,
        rmse=math.sqrt(mse),
        mae=math.fsum(weights * abs(p - y)),
        pearson=corr(y, p, weights),
        spearman=corr(ranks(y), ranks(p), weights),
        unweighted_rmse=math.sqrt(math.fsum((p - y) ** 2) / len(y)),
        unweighted_mae=math.fsum(abs(p - y)) / len(y),
        unweighted_pearson=corr(y, p, uniform),
        unweighted_spearman=corr(ranks(y), ranks(p), uniform),
        correlation_undefined_reason="constant target or prediction"
        if min(p) == max(p) or min(y) == max(y)
        else None,
        target_mean=math.fsum(weights * y),
        prediction_mean=math.fsum(weights * p),
        target_sd=target_sd,
        prediction_sd=prediction_sd,
        prediction_target_sd_ratio=prediction_sd / target_sd if target_sd else None,
        signed_change_accuracy=float(
            weights[nz] @ (np.sign(p[nz]) == np.sign(y[nz])) / weights[nz].sum()
        )
        if any(nz)
        else None,
        signed_change_weight=float(weights[nz].sum()),
        n_zero_targets=int((~nz).sum()),
        zero_target_weight=float(weights[~nz].sum()),
    )


pooled = {k: metric(list(range(len(rows))), v) for k, v in predictions.items()}
folds = {
    str(f): {
        k: metric([i for i, g in enumerate(group_ids) if outer(g) == f], v)
        for k, v in predictions.items()
    }
    for f in range(5)
}
check(
    dict(pooled=pooled, folds=folds),
    json.loads((ROOT / "results/metrics.json").read_text()),
)
errors = {}
for kind, values in predictions.items():
    errors[kind] = []
    for g in groups:
        ix = [i for i, group in enumerate(group_ids) if group == g]
        errors[kind].append(
            math.fsum(ws[ix] * (values[ix] - ys[ix]) ** 2) / math.fsum(ws[ix])
        )
check(
    dict(groups=groups, mse=errors),
    json.loads((ROOT / "results/component-errors.json").read_text()),
)
random = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"]))
draws = np.stack(
    [
        random.integers(0, len(groups), size=len(groups))
        for _ in range(CFG["bootstrap_draws"])
    ]
)
rmses = {k: np.sqrt(np.mean(np.asarray(v)[draws], axis=1)) for k, v in errors.items()}
gains = {k: 1 - rmses["boundary"] / rmses[k] for k in ("zero", "length", "sequence")}
intervals = {
    k: np.percentile(v, [2.5, 97.5], method="linear").tolist() for k, v in gains.items()
}
with np.load(ROOT / "results/bootstrap.npz", allow_pickle=False) as saved:
    assert np.array_equal(draws, saved["indices"])
    for prefix, collection in (("rmse_", rmses), ("gain_", gains)):
        for kind, value in collection.items():
            check(value.tolist(), saved[prefix + kind].tolist())
gain = {k: 1 - pooled["boundary"]["rmse"] / pooled[k]["rmse"] for k in gains}
spread = pooled["boundary"]["prediction_target_sd_ratio"]
improved = sum(
    folds[str(f)]["boundary"]["rmse"] < folds[str(f)]["sequence"]["rmse"]
    for f in range(5)
)
gates = dict(
    gain_over_all_controls=all(g >= 0.05 for g in gain.values()),
    spread=spread is not None and spread >= 0.25,
    four_of_five_folds=improved >= 4,
    bootstrap_lower_bound=intervals["sequence"][0] > 0,
)
decision = dict(
    gains=gain,
    prediction_target_sd_ratio=spread,
    improved_folds_vs_sequence=improved,
    paired_gain_intervals=intervals,
    gates=gates,
    prioritize_independent_confirmation=all(gates.values()),
    complete_APC_concentration_model=False,
    evidence_scope="Training-only development OOF; overlapping fit sets, no blind biological test.",
)
check(decision, json.loads((ROOT / "results/decision.json").read_text()))
result = dict(
    passed=True,
    independent_algorithm="literal startswith occurrences, augmented SVD least squares, manual ranks, per-draw PCG64 replay",
    numerical_values_checked=numeric_checks,
    max_absolute_difference=maximum_error,
    n_fits=240,
    n_inner_predictions=len(inner_predictions),
    n_oof_predictions=len(oof),
    n_endpoints=len(expected_context),
    n_bootstrap_draws=len(draws),
    no_production_core_import=True,
    no_outer_rows_materialized=True,
)
(ROOT / "results/replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
