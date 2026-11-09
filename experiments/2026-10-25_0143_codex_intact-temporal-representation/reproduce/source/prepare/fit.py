"""Nested fold-local direct fits and three-start exact processing controls."""

import hashlib
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
import torch

from core import (ROOT, common_fit, configuration, data_free_initial, dump, load_changes,
                  physical, physical_batch, physical_change, physical_objective, predictions,
                  ridge_fit, ridge_predict, sequences, weights)


def write_predictions(name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    pq.write_table(pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows]), ROOT / name)


def physical_fit(rows, arm, fold, start_number, specification):
    kind = "uniform" if arm == "physical_uniform" else "residue"
    alpha = .01 if arm == "physical_residue_penalized" else 0.
    batch, aa_index, amino_acids = physical_batch(rows)
    initial = data_free_initial(kind, len(amino_acids), **specification)
    output = ROOT / f"results/physical/fold{fold}-{arm}-start{start_number}"
    output.mkdir(parents=True)
    history, accepted = [], [initial.copy()]
    start_time = time.monotonic()

    def objective(x):
        theta = torch.tensor(x, requires_grad=True)
        value, mse, penalty = physical_objective(theta, kind, batch, aa_index, alpha)
        gradient = torch.autograd.grad(value, theta)[0].detach().numpy()
        assert np.isfinite(gradient).all() and np.isfinite(value.item())
        history.append(np.r_[time.monotonic() - start_time, value.item(), mse.item(), penalty.item(), x, gradient])
        return value.item(), gradient

    try:
        result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=[(-30., 30.)] * len(initial),
                          callback=lambda x: accepted.append(x.copy()),
                          options={"maxiter": 1500, "maxfun": 3000, "maxls": 40, "gtol": 1e-9, "ftol": 1e-13})
    except Exception as error:
        np.savez_compressed(output / "failed-history.npz", evaluations=np.array(history), accepted=np.array(accepted))
        (output / "failure.json").write_text(json.dumps({"error": repr(error), "training_sequences": sequences(rows)}, indent=2) + "\n")
        raise
    elapsed = time.monotonic() - start_time
    theta = torch.tensor(result.x)
    value, mse, penalty = physical_objective(theta, kind, batch, aa_index, alpha)
    _, tau, rates = physical.decode(theta, kind, aa_index)
    np.savez_compressed(output / "optimizer-history.npz", evaluations=np.array(history), accepted=np.array(accepted),
                        initial=initial, final=result.x, gradient=result.jac)
    train_prediction = physical_change(theta, kind, batch, aa_index).numpy()
    write_predictions(str((output / "training-predictions.parquet").relative_to(ROOT)),
                      predictions(rows, train_prediction, outer_fit=fold, role="training", arm=arm,
                                  start_number=start_number))
    summary = {"arm": arm, "kind": kind, "alpha": alpha, "outer_fold": fold, "start_number": start_number,
               "initialization": specification, "training_sequences": sequences(rows), "n_training_observations": len(rows),
               "success": bool(result.success), "message": str(result.message), "seconds": elapsed,
               "iterations": int(result.nit), "function_evaluations": int(result.nfev),
               "objective": value.item(), "mse": mse.item(), "penalty": penalty.item(),
               "raw_parameters": result.x.tolist(), "raw_gradient": result.jac.tolist(),
               "tau_minutes": tau.item(), "rates_by_category": rates.tolist(),
               "all_parameters_changed": bool(np.all(result.x != initial)),
               "gradient_absolute_totals": np.abs(np.array(history)[:, 4 + len(initial):]).sum(0).tolist(),
               "raw_boundary_coordinates": np.flatnonzero(np.abs(result.x) >= 30 - 1e-7).tolist()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    for name in ("prepare", "qualify"):
        assert json.loads((ROOT / f"results/{name}-resource.json").read_text())["passed"]
    config = configuration()
    rows = load_changes()
    output_rows, inner_rows, all_fits = [], [], []
    direct_records = []
    for outer in range(5):
        train = [r for r in rows if r["outer_fold"] != outer]
        validation = [r for r in rows if r["outer_fold"] == outer]
        assert not set(sequences(train)) & set(sequences(validation))
        common = common_fit(train)
        direct_records.append({"outer_fold": outer, "arm": "shared_slope", "slope": common,
                               "training_sequences": sequences(train)})
        for role, query in (("training", train), ("validation", validation)):
            output_rows.extend(predictions(query, np.zeros(len(query)), outer_fit=outer, role=role, arm="persistence"))
            output_rows.extend(predictions(query, common * np.array([r["elapsed"] for r in query]),
                                           outer_fit=outer, role=role, arm="shared_slope"))
        for representation in ("length", "sequence"):
            candidate_predictions = {alpha: [] for alpha in config["alpha_grid"]}
            for inner in range(3):
                inner_train = [r for r in train if r["inner_fold"] != inner]
                inner_validation = [r for r in train if r["inner_fold"] == inner]
                assert not set(sequences(inner_train)) & set(sequences(inner_validation))
                for alpha in config["alpha_grid"]:
                    fitted = ridge_fit(inner_train, representation, alpha)
                    values = ridge_predict(inner_validation, fitted)
                    predicted = predictions(inner_validation, values, outer_fit=outer, inner_fit=inner,
                                            role="inner_validation", arm=representation, alpha=alpha)
                    candidate_predictions[alpha].extend(predicted)
                    inner_rows.extend(predicted)
                    direct_records.append({"outer_fold": outer, "inner_fold": inner, "arm": representation,
                                           "role": "inner", **fitted})
            scores = {}
            for alpha, predicted in candidate_predictions.items():
                assert set(r["assignment_id"] for r in predicted) == set(r["assignment_id"] for r in train)
                assert len(predicted) == len(train)
                scores[alpha] = float(weights(predicted) @ np.array([(r["prediction"] - r["target"]) ** 2 for r in predicted]))
            selected_alpha = min(config["alpha_grid"], key=lambda a: scores[a])
            fitted = ridge_fit(train, representation, selected_alpha)
            direct_records.append({"outer_fold": outer, "arm": representation, "role": "selected_outer",
                                   "inner_mse": scores, **fitted})
            for role, query in (("training", train), ("validation", validation)):
                output_rows.extend(predictions(query, ridge_predict(query, fitted), outer_fit=outer, role=role,
                                               arm=representation, selected_alpha=selected_alpha))
        for arm in ("physical_uniform", "physical_residue_penalized", "physical_residue_unpenalized"):
            starts = [physical_fit(train, arm, outer, number, specification)
                      for number, specification in enumerate(config["starts"])]
            all_fits.extend(starts)
            selected = min(starts, key=lambda f: f["objective"])
            assert np.isfinite(selected["objective"])
            for role, query in (("training", train), ("validation", validation)):
                batch, aa_index, _ = physical_batch(query)
                theta = torch.tensor(selected["raw_parameters"])
                values = physical_change(theta, selected["kind"], batch, aa_index).numpy()
                output_rows.extend(predictions(query, values, outer_fit=outer, role=role, arm=arm,
                                               selected_start=selected["start_number"], optimizer_success=selected["success"]))
            print(json.dumps({"outer_fold": outer, "arm": arm, "selected_start": selected["start_number"],
                              "training_mse": selected["mse"], "success": selected["success"]}), flush=True)
    write_predictions("results/inner-predictions.parquet", inner_rows)
    write_predictions("results/predictions.parquet", output_rows)
    dump("results/direct-fits.json", direct_records)
    dump("results/physical-fits.json", all_fits)
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                ("results/inner-predictions.parquet", "results/predictions.parquet", "results/direct-fits.json", "results/physical-fits.json")}
    dump("results/fit-summary.json", {"n_physical_starts": len(all_fits), "n_prediction_rows": len(output_rows),
         "n_inner_prediction_rows": len(inner_rows), "n_direct_fit_records": len(direct_records),
         "output_hashes": manifest, "initializer_uses_labels": False,
         "new_heldout_biological_test": False, "scope": "nested_development_CV_in_original_training"})


if __name__ == "__main__":
    main()
