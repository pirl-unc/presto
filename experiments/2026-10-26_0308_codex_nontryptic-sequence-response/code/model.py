"""Fixed sequence-response models; all scaling and selection use training rows."""

import numpy as np
from scipy.stats import rankdata

from common import ids_digest

AA = "ACDEFGHIKLMNPQRSTVWY"
FEATURES = ["frequency_" + aa for aa in AA] + ["sequence_length"]


def sequence_features(sequence):
    if not sequence or set(sequence) - set(AA):
        raise ValueError("Unsupported sequence")
    return [sequence.count(aa) / len(sequence) for aa in AA] + [len(sequence)]


def weights(rows):
    sequences = sorted({r["sequence"] for r in rows})
    if not sequences:
        raise ValueError("Empty training population")
    doses = {s: {r["x"] for r in rows if r["sequence"] == s} for s in sequences}
    result = []
    for r in rows:
        n_slots = sum(q["sequence"] == r["sequence"] and q["x"] == r["x"] for q in rows)
        result.append(1 / (len(sequences) * len(doses[r["sequence"]]) * n_slots))
    return np.asarray(result, dtype=float)


def fit(rows, arm, alpha=0.0):
    if arm not in {"unit", "slope", "sequence"}:
        raise ValueError(arm)
    if arm == "sequence" and alpha <= 0:
        raise ValueError("Sequence ridge requires positive alpha")
    x = np.asarray([r["x"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(set(x)) < 2:
        raise ValueError("Need finite observations and at least two nominal doses")
    w = weights(rows)
    model = {
        "arm": arm,
        "alpha": float(alpha),
        "training_ids": sorted(r["measurement_id"] for r in rows),
        "training_genes": sorted({r["gene"] for r in rows}),
        "training_sequences": sorted({r["sequence"] for r in rows}),
        "active_features": [],
        "feature_means": [],
        "feature_scales": [],
    }
    model["training_ids_sha256"] = ids_digest(model["training_ids"])
    columns = [np.ones(len(rows))]
    if arm == "unit":
        y = y - x
    else:
        columns.append(x)
    if arm == "sequence":
        unique = np.asarray([sequence_features(s) for s in model["training_sequences"]])
        mean, scale = unique.mean(axis=0), unique.std(axis=0)
        active = np.flatnonzero(scale > 1e-12)
        z = np.asarray([sequence_features(r["sequence"]) for r in rows])
        z = (z[:, active] - mean[active]) / scale[active]
        columns.extend(z.T)
        model.update(
            active_features=active.tolist(),
            feature_means=mean[active].tolist(),
            feature_scales=scale[active].tolist(),
        )
    design = np.column_stack(columns)
    penalty = np.zeros(design.shape[1])
    if arm == "sequence":
        penalty[2:] = alpha
    matrix = design.T @ (w[:, None] * design) + np.diag(penalty)
    model["coefficients"] = np.linalg.solve(matrix, design.T @ (w * y)).tolist()
    model["design_rank"] = int(np.linalg.matrix_rank(design))
    model["regularized_rank"] = int(np.linalg.matrix_rank(matrix))
    model["n_parameters"] = len(model["coefficients"])
    assert model["regularized_rank"] == model["n_parameters"]
    model["dose_slope"] = 1.0 if arm == "unit" else model["coefficients"][1]
    return model


def predict(model, rows):
    x = np.asarray([r["x"] for r in rows])
    columns = [np.ones(len(rows))]
    if model["arm"] != "unit":
        columns.append(x)
    if model["arm"] == "sequence":
        raw = np.asarray([sequence_features(r["sequence"]) for r in rows])
        z = (raw[:, model["active_features"]] - model["feature_means"]) / model["feature_scales"]
        columns.extend(z.T)
    value = np.column_stack(columns) @ model["coefficients"]
    return value + x if model["arm"] == "unit" else value


def equal_peptide_mse(rows, predictions):
    y = np.asarray([r["y"] for r in rows])
    squared = (np.asarray(predictions) - y) ** 2
    return float(
        np.mean(
            [
                np.mean([v for r, v in zip(rows, squared, strict=True) if r["sequence"] == s])
                for s in sorted({r["sequence"] for r in rows})
            ]
        )
    )


def select_alpha(rows, alphas):
    genes = sorted({r["gene"] for r in rows})
    if len(genes) < 3:
        raise ValueError("Insufficient precursor groups for nested selection")
    models, predictions, losses = [], [], {}
    for alpha in alphas:
        selected_rows, selected_predictions = [], []
        for held in genes:
            train = [r for r in rows if r["gene"] != held]
            val = [r for r in rows if r["gene"] == held]
            assert not {r["sequence"] for r in train} & {r["sequence"] for r in val}
            m = fit(train, "sequence", alpha)
            m.update(inner_gene=held, local_model_id=len(models))
            models.append(m)
            values = predict(m, val)
            for r, value in zip(val, values, strict=True):
                predictions.append(
                    {
                        "measurement_id": r["measurement_id"],
                        "prediction": float(value),
                        "target": r["y"],
                        "inner_gene": held,
                        "alpha": alpha,
                        "local_model_id": m["local_model_id"],
                    }
                )
            selected_rows.extend(val)
            selected_predictions.extend(values)
        losses[alpha] = equal_peptide_mse(selected_rows, selected_predictions)
    # A relative/absolute numerical tie uses the predeclared stronger penalty.
    minimum = min(losses.values())
    chosen = max(
        a for a, loss in losses.items() if np.isclose(loss, minimum, rtol=1e-12, atol=1e-12)
    )
    return chosen, losses, models, predictions


def metrics(rows, predictions):
    y = np.asarray([r["y"] for r in rows])
    p = np.asarray(predictions)
    x = np.asarray([r["x"] for r in rows])
    error = p - y

    def corr(left, right):
        if np.std(left) < 1e-12 or np.std(right) < 1e-12:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    result = {
        "n_rows": len(rows),
        "n_peptides": len({r["sequence"] for r in rows}),
        "pooled_mse": float(np.mean(error**2)),
        "pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "pooled_mae": float(np.mean(np.abs(error))),
        "equal_peptide_mse": equal_peptide_mse(rows, p),
        "equal_peptide_rmse": float(np.sqrt(equal_peptide_mse(rows, p))),
        "equal_peptide_mae": float(
            np.mean(
                [
                    np.mean(
                        [abs(e) for r, e in zip(rows, error, strict=True) if r["sequence"] == s]
                    )
                    for s in sorted({r["sequence"] for r in rows})
                ]
            )
        ),
        "pearson": corr(y, p),
        "spearman": corr(rankdata(y), rankdata(p)),
        "target_sd": float(np.std(y)),
        "prediction_sd": float(np.std(p)),
        "dose_adjusted_target_sd": float(np.std(y - x)),
        "dose_adjusted_prediction_sd": float(np.std(p - x)),
        "dose_adjusted_pearson": corr(y - x, p - x),
        "undefined_correlation_policy": "null for a vector with SD below1e-12",
    }
    yc, pc = y.copy(), p.copy()
    for s in {r["sequence"] for r in rows}:
        mask = np.asarray([r["sequence"] == s for r in rows])
        yc[mask] -= y[mask].mean()
        pc[mask] -= p[mask].mean()
    result.update(
        within_peptide_target_sd=float(yc.std()),
        within_peptide_prediction_sd=float(pc.std()),
        within_peptide_rmse=float(np.sqrt(np.mean((pc - yc) ** 2))),
        within_peptide_pearson=corr(yc, pc),
    )
    return result
