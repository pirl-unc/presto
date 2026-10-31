"""Independent raw-cell target, all-pair fit and two-space result replay."""

from collections import Counter, defaultdict
import hashlib, json, math
from itertools import combinations
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PKG = Path(CFG["package"])
PRIOR = Path(CFG["reference_family"])
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


# Reconstruct both row spaces directly from admitted original cells and covariates.
original = sorted(
    read("typed/examples.parquet", [("split", "=", "train")]),
    key=lambda r: r["example_id"],
)
assert len(original) == 158 and all(r["split"] == "train" for r in original)
contributors = read(
    "typed/contributors.parquet",
    [("example_id", "in", [r["example_id"] for r in original])],
)
quantities = read(
    "source_tables/quantities.parquet",
    [("measurement_id", "in", sorted({c["measurement_id"] for c in contributors}))],
)
qmap = {q["measurement_id"]: q for q in quantities}
assert len(qmap) == len(quantities) == 1284 and len(contributors) == 1896
acqs = {
    a["acquisition_id"]: a
    for a in read(
        "source_tables/acquisitions.parquet",
        [("acquisition_id", "in", sorted({c["acquisition_id"] for c in contributors}))],
    )
}
cells = defaultdict(dict)
for c in contributors:
    q = qmap[c["measurement_id"]]
    assert acqs[q["acquisition_id"]]["sample_label_raw"] == c["sample_label_raw"]
    for key in (
        "quantitative_raw",
        "quantitative_value",
        "peptide_id",
        "source_record_id",
        "acquisition_id",
    ):
        assert c[key] == q[key]
    value = float(q["quantitative_raw"])
    assert value == q["quantitative_value"] > 0
    key = c["peptide_id"]
    label = c["sample_label_raw"]
    if label in cells[key]:
        assert cells[key][label] == value
    cells[key][label] = value
labels = ["Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3"]
assert len(cells) == 214 and all(set(v) == set(labels) for v in cells.values())
parents = {
    r["parent_sequence_sha256"]: r
    for r in read("typed/parents.parquet", [("split", "=", "train")])
}
by_parent = defaultdict(list)
for r in original:
    by_parent[r["parent_sequence_sha256"]].append(r)
groupcounts = Counter(rows[0]["dependency_group_id"] for rows in by_parent.values())
assert len(groupcounts) == 55 and len(parents) == len(by_parent) == 56
old_context = json.loads((PRIOR / "results/endpoint-contexts.json").read_text())
features = {}
for key, record in old_context.items():
    parent = parents[record["parent_sequence_sha256"]]["parent_sequence"]
    assert (
        hashlib.sha256(parent.encode()).hexdigest() == record["parent_sequence_sha256"]
    )
    vector, spans = endpoint(parent, record["peptide_sequence"])
    check(vector.tolist(), record["endpoint_features"])
    check(spans, record["occurrences"])
    features[key] = vector


def raw_target(a, b):
    ratios = [math.log10(cells[a][label] / cells[b][label]) for label in labels]
    return math.fsum(ratios[3:]) / 3 - math.fsum(ratios[:3]) / 3


def difference(parent, a, b):
    delta = features[a + "|" + parent] - features[b + "|" + parent]
    return {
        "length": delta[:1].tolist(),
        "sequence": delta[:61].tolist(),
        "boundary": delta.tolist(),
    }


reference = []
for r in original:
    check(
        raw_target(r["peptide_id"], r["reference_peptide_id"]),
        float(r["target_log10_ratio_change"]),
    )
    reference.append(
        {
            **r,
            "features": difference(
                r["parent_sequence_sha256"], r["peptide_id"], r["reference_peptide_id"]
            ),
        }
    )
paired = []
support_parents = []
strata = {
    key: dict(
        n_parents=0,
        n_endpoints=0,
        n_pairs=0,
        component_parent_weight=0.0,
        hierarchical_endpoint_weight=0.0,
    )
    for key in ("n2", "n3", "n4plus")
}
independent_gram_max = 0.0
for digest, records in sorted(by_parent.items()):
    refs = {r["reference_peptide_id"] for r in records}
    assert len(refs) == 1
    ref = next(iter(refs))
    ids = sorted({ref, *[r["peptide_id"] for r in records]})
    n = len(ids)
    group = records[0]["dependency_group_id"]
    mass = 1 / (55 * groupcounts[group])
    n_pairs = n * (n - 1) // 2
    assert all(
        r["dependency_group_id"] == group and abs(r["weight"] - mass / (n - 1)) < 1e-14
        for r in records
    )
    support_parents.append(
        dict(
            parent_sequence_sha256=digest,
            dependency_group_id=group,
            reference_peptide_id=ref,
            endpoint_ids=ids,
            n_endpoints=n,
            n_pairs=n_pairs,
            parent_weight=mass,
        )
    )
    bucket = strata["n2" if n == 2 else "n3" if n == 3 else "n4plus"]
    bucket["n_parents"] += 1
    bucket["n_endpoints"] += n
    bucket["n_pairs"] += n_pairs
    bucket["component_parent_weight"] += mass
    bucket["hierarchical_endpoint_weight"] += n * (mass / n)
    x = np.array([features[pid + "|" + digest] for pid in ids])
    y = np.array([raw_target(pid, ref) for pid in ids])
    z = np.column_stack((x, y))
    center = z - z.mean(0)
    pair_matrix = np.array([z[i] - z[j] for i, j in combinations(range(n), 2)])
    gram = np.einsum("ij,ik->jk", pair_matrix, pair_matrix) / n_pairs
    equivalent = 2 / (n - 1) * np.einsum("ij,ik->jk", center, center)
    error = float(np.max(np.abs(gram - equivalent)))
    independent_gram_max = max(error, independent_gram_max)
    assert error < 2e-10
    for a, b in combinations(ids, 2):
        identifier = hashlib.sha256(
            ("native-pair-v1|" + digest + "|" + a + "|" + b).encode()
        ).hexdigest()
        target = raw_target(a, b)
        check(float(target), float(raw_target(a, ref) - raw_target(b, ref)))
        paired.append(
            dict(
                example_id=identifier,
                peptide_id=a,
                reference_peptide_id=b,
                parent_sequence_sha256=digest,
                dependency_group_id=group,
                split="train",
                weight=mass / n_pairs,
                target_log10_ratio_change=target,
                features=difference(digest, a, b),
            )
        )
paired.sort(key=lambda r: r["example_id"])
data = {"reference": reference, "all_pairs": paired}
for key, rows in data.items():
    check(rows, json.loads((ROOT / "results" / f"{key}-rows.json").read_text()))
support = json.loads((ROOT / "results/support.json").read_text())
for record in strata.values():
    record["unweighted_parent_fraction"] = record["n_parents"] / 56
    record["unweighted_endpoint_fraction"] = record["n_endpoints"] / 214
check(strata, support["strata"])
check(support_parents, support["parents"])
assert support["n_pairs"] == len(paired) and support["n_reference_degrees"] == 158
for record in json.loads((ROOT / "results/algebra.json").read_text()):
    assert (
        record["maximum_gram_identity_error"] < 2e-10
        and record["maximum_gradient_identity_error"] < 2e-10
    )

# Fit only new all-pair models through an alternative augmented SVD algorithm.
train_rows = paired
train_x = {
    kind: np.array([r["features"][kind] for r in train_rows])
    for kind in ("length", "sequence", "boundary")
}
train_y = np.array([r["target_log10_ratio_change"] for r in train_rows])
train_w = np.array([r["weight"] for r in train_rows])
train_groups = [r["dependency_group_id"] for r in train_rows]


def verify_fit(indices, kind, alpha, stored):
    x, y, w = train_x[kind][indices], train_y[indices], train_w[indices]
    w = w / w.sum()
    scales = np.sqrt(np.einsum("i,ij,ij->j", w, x, x))
    active = np.flatnonzero(scales != 0)
    z = x[:, active] / scales[active]
    system = np.vstack(
        (np.sqrt(w)[:, None] * z, math.sqrt(alpha) * np.eye(len(active)))
    )
    rhs = np.r_[np.sqrt(w) * y, np.zeros(len(active))]
    beta = np.linalg.lstsq(system, rhs, rcond=None)[0]
    coeff = np.zeros(x.shape[1])
    coeff[active] = beta
    mse = float(w @ (z @ beta - y) ** 2)
    penalty = float(alpha * sum(beta**2))
    expected = dict(
        representation=kind,
        alpha=float(alpha),
        scales=scales.tolist(),
        active_coordinates=active.tolist(),
        coefficients=coeff.tolist(),
        n_raw_features=x.shape[1],
        n_solved_coordinates=len(active),
        design_rank=int(np.linalg.matrix_rank(z)) if len(active) else 0,
        n_nonzero_coefficients=int(np.count_nonzero(beta)),
        stationarity_max_abs=stored["stationarity_max_abs"],
        training_mse=mse,
        penalty=penalty,
        training_objective=mse + penalty,
        training_example_ids=[train_rows[i]["example_id"] for i in indices],
        training_components=sorted({train_groups[i] for i in indices}),
    )
    check(expected, stored)
    original_beta = np.array(stored["coefficients"])[active]
    gradient = z.T @ (w * (z @ original_beta - y)) + alpha * original_beta
    assert (
        np.max(np.abs(gradient), initial=0) < 1e-10
        and stored["stationarity_max_abs"] < 1e-10
    )
    return active, scales, beta


def new_predictions(rows, kind, model):
    active, scales, beta = model
    x = np.array([r["features"][kind] for r in rows])
    return (x[:, active] / scales[active]) @ beta


def original_predictions(rows, stored):
    if stored["representation"] == "zero":
        return np.zeros(len(rows))
    kind = stored["representation"]
    active = np.array(stored["active_coordinates"])
    x = np.array([r["features"][kind] for r in rows])
    return (x[:, active] / np.array(stored["scales"])[active]) @ np.array(
        stored["coefficients"]
    )[active]


records = json.loads((ROOT / "results/inner-fits.json").read_text())
assert len(records) == 225
inner_pred = pq.read_table(ROOT / "results/inner-predictions.parquet").to_pylist()
lookup = {
    (r["outer_fold"], r["kind"], r["alpha"], r["fold"], r["example_id"]): r
    for r in inner_pred
}
assert len(lookup) == len(inner_pred)
losses = defaultdict(list)
for record in records:
    f, k, a, j = record["outer_fold"], record["kind"], record["alpha"], record["fold"]
    fit_indices = [
        i for i, g in enumerate(train_groups) if outer(g) != f and inner(g) != j
    ]
    valid = [i for i, g in enumerate(train_groups) if outer(g) != f and inner(g) == j]
    fitted = verify_fit(fit_indices, k, a, record["fitted"])
    pred = new_predictions([train_rows[i] for i in valid], k, fitted)
    weight = train_w[valid] / train_w[valid].sum()
    loss = float(weight @ (pred - train_y[valid]) ** 2)
    check(loss, record["loss"])
    losses[f, k, a].append(loss)
    for i, value, w in zip(valid, pred, weight):
        check(
            dict(
                outer_fold=f,
                example_id=train_rows[i]["example_id"],
                kind=k,
                alpha=a,
                fold=j,
                target=float(train_y[i]),
                prediction=float(value),
                weight=float(w),
            ),
            lookup.pop((f, k, a, j, train_rows[i]["example_id"])),
        )
assert not lookup
models = json.loads((ROOT / "results/models.json").read_text())
assert len(models) == 20
original_models = {
    (r["outer_fold"], r["kind"]): r["fitted"]
    for r in json.loads((PRIOR / "results/models.json").read_text())
}
old = {
    (r["example_id"], r["kind"]): r
    for r in pq.read_table(
        PRIOR / "results/development-oof-predictions.parquet"
    ).to_pylist()
}
oof = pq.read_table(ROOT / "results/development-oof-predictions.parquet").to_pylist()
lookup = {
    (r["evaluation_space"], r["training_objective"], r["kind"], r["example_id"]): r
    for r in oof
}
assert len(lookup) == len(oof)
outputs = {
    space: {
        objective + ":" + kind: np.empty(len(rows))
        for objective in ("reference", "all_pairs")
        for kind in CFG["representations"]
    }
    for space, rows in data.items()
}
for record in models:
    f, kind = record["outer_fold"], record["kind"]
    fit_indices = [i for i, g in enumerate(train_groups) if outer(g) != f]
    if kind != "zero":
        means = {a: float(np.mean(losses[f, kind, a])) for a in CFG["alpha_grid"]}
        alpha = min(CFG["alpha_grid"], key=lambda a: (means[a], -a))
        assert alpha == record["fitted"]["alpha"]
        check({str(k): v for k, v in means.items()}, record["selection_mean_mse"])
        fitted = verify_fit(fit_indices, kind, alpha, record["fitted"])
    for space, rows in data.items():
        valid = [i for i, r in enumerate(rows) if outer(r["dependency_group_id"]) == f]
        omitted = [rows[i] for i in valid]
        for objective in ("reference", "all_pairs"):
            values = (
                original_predictions(omitted, original_models[f, kind])
                if objective == "reference"
                else np.zeros(len(valid))
                if kind == "zero"
                else new_predictions(omitted, kind, fitted)
            )
            outputs[space][objective + ":" + kind][valid] = values
            for i, value in zip(valid, values):
                row = rows[i]
                check(
                    dict(
                        training_objective=objective,
                        evaluation_space=space,
                        outer_fold=f,
                        kind=kind,
                        example_id=row["example_id"],
                        dependency_group_id=row["dependency_group_id"],
                        parent_sequence_sha256=row["parent_sequence_sha256"],
                        original_split="train",
                        weight=row["weight"],
                        target=row["target_log10_ratio_change"],
                        prediction=float(value),
                    ),
                    lookup.pop((space, objective, kind, row["example_id"])),
                )
                if objective == space == "reference":
                    check(
                        float(value), old.pop((row["example_id"], kind))["prediction"]
                    )
assert not lookup and not old


def metric(rows, p):
    y = np.array([r["target_log10_ratio_change"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    w = w / w.sum()
    u = np.ones(len(y)) / len(y)
    mse = math.fsum(w * (p - y) ** 2)
    tsd = math.sqrt(math.fsum(w * (y - math.fsum(w * y)) ** 2))
    psd = (
        0.0
        if min(p) == max(p)
        else math.sqrt(math.fsum(w * (p - math.fsum(w * p)) ** 2))
    )
    nz = y != 0
    return dict(
        n_contrasts=len(rows),
        n_components=len({r["dependency_group_id"] for r in rows}),
        loss=mse,
        mse=mse,
        rmse=math.sqrt(mse),
        mae=math.fsum(w * abs(p - y)),
        pearson=corr(y, p, w),
        spearman=corr(ranks(y), ranks(p), w),
        unweighted_rmse=math.sqrt(math.fsum((p - y) ** 2) / len(y)),
        unweighted_mae=math.fsum(abs(p - y)) / len(y),
        unweighted_pearson=corr(y, p, u),
        unweighted_spearman=corr(ranks(y), ranks(p), u),
        correlation_undefined_reason="constant target or prediction"
        if min(p) == max(p) or min(y) == max(y)
        else None,
        target_mean=math.fsum(w * y),
        prediction_mean=math.fsum(w * p),
        target_sd=tsd,
        prediction_sd=psd,
        prediction_target_sd_ratio=psd / tsd if tsd else None,
        signed_change_accuracy=float(
            w[nz] @ (np.sign(p[nz]) == np.sign(y[nz])) / w[nz].sum()
        )
        if any(nz)
        else None,
        signed_change_weight=float(w[nz].sum()),
        n_zero_targets=int((~nz).sum()),
        zero_target_weight=float(w[~nz].sum()),
    )


scores = {}
components = {}
intervals = {}
references = [
    "all_pairs:zero",
    "all_pairs:length",
    "all_pairs:sequence",
    "reference:boundary",
]
primary = "all_pairs:boundary"
with np.load(ROOT / "results/bootstrap.npz", allow_pickle=False) as saved:
    for space, rows in data.items():
        values = outputs[space]
        pooled = {k: metric(rows, v) for k, v in values.items()}
        folds = {}
        for f in range(5):
            ix = [i for i, r in enumerate(rows) if outer(r["dependency_group_id"]) == f]
            folds[str(f)] = {
                k: metric([rows[i] for i in ix], v[ix]) for k, v in values.items()
            }
        scores[space] = dict(pooled=pooled, folds=folds)
        groups = sorted({r["dependency_group_id"] for r in rows})
        errors = {}
        for key, pred in values.items():
            errors[key] = []
            for g in groups:
                ix = [i for i, r in enumerate(rows) if r["dependency_group_id"] == g]
                mass = math.fsum(rows[i]["weight"] for i in ix)
                errors[key].append(
                    math.fsum(
                        rows[i]["weight"]
                        * (pred[i] - rows[i]["target_log10_ratio_change"]) ** 2
                        for i in ix
                    )
                    / mass
                )
        components[space] = dict(groups=groups, mse=errors)
        random = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"]))
        draws = np.stack(
            [
                random.integers(0, len(groups), size=len(groups))
                for _ in range(CFG["bootstrap_draws"])
            ]
        )
        assert np.array_equal(draws, saved[space + "__indices"])
        rmse = {k: np.sqrt(np.array(v)[draws].mean(1)) for k, v in errors.items()}
        gains = {k: 1 - rmse[primary] / rmse[k] for k in references}
        intervals[space] = {
            k: np.percentile(v, [2.5, 97.5], method="linear").tolist()
            for k, v in gains.items()
        }
        for prefix, collection in (("rmse", rmse), ("gain", gains)):
            for key, value in collection.items():
                check(
                    value.tolist(), saved[space + "__" + prefix + "__" + key].tolist()
                )
check(scores, json.loads((ROOT / "results/metrics.json").read_text()))
check(components, json.loads((ROOT / "results/component-errors.json").read_text()))
check(intervals, json.loads((ROOT / "results/intervals.json").read_text()))
pooled = scores["all_pairs"]["pooled"]
folds = scores["all_pairs"]["folds"]
gain = {k: 1 - pooled[primary]["rmse"] / pooled[k]["rmse"] for k in references}
spread = pooled[primary]["prediction_target_sd_ratio"]
improved = sum(
    folds[str(f)][primary]["rmse"] < folds[str(f)]["reference:boundary"]["rmse"]
    for f in range(5)
)
gates = dict(
    gain_over_all_matched_controls=all(v >= 0.05 for v in gain.values()),
    spread=spread is not None and spread >= 0.25,
    four_of_five_folds=improved >= 4,
    positive_matched_bootstrap_lower_bound=intervals["all_pairs"]["reference:boundary"][
        0
    ]
    > 0,
)
passed = all(gates.values())
decision = dict(
    gains=gain,
    prediction_target_sd_ratio=spread,
    improved_folds_vs_reference_objective=improved,
    intervals=intervals["all_pairs"],
    gates=gates,
    prioritize_independent_confirmation=passed,
    stop_response_only_feature_optimizer_expansion=not passed,
    complete_APC_concentration_model=False,
    evidence_scope="Training-component development; pairs are dependent, outer cohorts remain closed.",
)
check(decision, json.loads((ROOT / "results/decision.json").read_text()))
receipt = dict(
    passed=True,
    n_raw_cells=len(quantities),
    n_reference_rows=len(reference),
    n_all_pairs=len(paired),
    n_new_fits=240,
    n_reused_original_controls=20,
    n_inner_predictions=len(inner_pred),
    n_oof_predictions=len(oof),
    n_bootstrap_draws_per_space=2000,
    numeric_values_checked=numeric_checks,
    max_absolute_difference=maximum_error,
    independent_gram_error=independent_gram_max,
    no_production_core_import=True,
    no_original_model_refit=True,
    no_outer_rows_materialized=True,
)
(ROOT / "results/replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
