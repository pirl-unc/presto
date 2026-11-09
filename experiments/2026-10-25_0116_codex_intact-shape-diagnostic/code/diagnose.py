"""Training-only shape restrictions and fixed-objective optimizer comparison."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.optimize import minimize
import torch

import frozen_model as model
from shapes import fit_shape, independent_slsqp, pava, qualify

ROOT = Path(__file__).resolve().parents[1]


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_train():
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    prior = Path(config["prior_family"])
    rows = pq.read_table(prior / "results/data/train/covariates.parquet").to_pylist()
    labels = pq.read_table(prior / "results/data/train/outcomes.parquet").to_pylist()
    assert [r["assignment_id"] for r in rows] == [r["assignment_id"] for r in labels]
    sequences = sorted({r["peptide_sequence"] for r in rows})
    parent = rows[0]["parent_sequence"]
    assert all(r["parent_sequence"] == parent for r in rows)
    amino_acids = sorted(set(parent[:-1]))
    aa_index = torch.tensor([amino_acids.index(a) for a in parent[:-1]])
    train = {
        "start": torch.tensor([r["start_1based"] - 1 for r in rows]),
        "end": torch.tensor([r["end_1based"] for r in rows]),
        "time": torch.tensor([r["time_minutes"] for r in rows], dtype=torch.float64),
        "weight": torch.tensor([r["weight"] for r in rows]),
        "target": torch.tensor([r["target"] for r in labels]),
        "sequence_index": torch.tensor([sequences.index(r["peptide_sequence"]) for r in rows]),
        "n_sequences": len(sequences),
    }
    return prior, rows, train, aa_index, amino_acids, sequences


def production_shape_checks(train, aa_index, states):
    results = []
    for kind, name, theta in states:
        batch = {**train, "time": train["time"].clone().requires_grad_()}
        values = model.component(theta, kind, batch, aa_index)
        first = torch.autograd.grad(values.sum(), batch["time"], create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), batch["time"])[0]
        rates, tau, _ = model.decode(theta, kind, aa_index)
        internal, boundary = model.masks(batch["start"], batch["end"], len(aa_index) + 1)
        exposure = batch["time"] + tau
        h = exposure[:, None] * rates
        analytic_first = (((~internal) * rates).sum(1) +
                          (boundary * rates / torch.expm1(h)).sum(1)) / math.log(10)
        analytic_second = -(boundary * rates.square() * torch.exp(h) /
                            torch.expm1(h).square()).sum(1) / math.log(10)
        error = max(float((first - analytic_first).abs().max().detach()),
                    float((second - analytic_second).abs().max().detach()))
        assert error <= 1e-8
        assert first.min().item() >= -1e-9 and second.max().item() <= 1e-9
        results.append({"kind": kind, "state": name, "analytic_derivative_max_error": error,
                        "minimum_first_derivative": first.min().item(),
                        "maximum_second_derivative": second.max().item()})
    return results


def main():
    # Actual solver fixtures precede any biological curve fitting.
    fixtures = qualify()
    prior, rows, train, aa_index, amino_acids, sequences = load_train()
    states = []
    original = {}
    for kind in ("uniform", "residue"):
        path = prior / f"results/fits/fit-{kind}-exact-0-42"
        original[kind] = json.loads((path / "summary.json").read_text())
        final = torch.tensor(original[kind]["raw_parameters"])
        initial = model.initial(kind, len(amino_acids)).detach()
        mixed = initial + .4 * torch.sin(torch.arange(len(initial)))
        for name, theta in (("initial", initial), ("adam_final", final), ("mixed", mixed)):
            states.append((kind, name, theta))
    shape_checks = production_shape_checks(train, aa_index, states)
    save("results/qualification.json", {"passed": True, "solver_fixtures": fixtures,
                                        "production_shape": shape_checks})

    y = train["target"].numpy()
    w = train["weight"].numpy()
    t = train["time"].numpy()
    seq = train["sequence_index"].numpy()
    fitted = {name: np.zeros(len(y)) for name in ("free", "increasing", "increasing_concave")}
    curve_rows = []
    max_shape_error = 0.
    negative_adjacent = total_adjacent = negative_slopes = 0
    for i, sequence in enumerate(sequences):
        index = np.flatnonzero(seq == i)
        times = np.unique(t[index])
        mass = np.array([w[index][t[index] == time].sum() for time in times])
        means = np.array([np.average(y[index][t[index] == time], weights=w[index][t[index] == time]) for time in times])
        mono, mono_info = fit_shape(means, mass / mass.sum(), times)
        concave, concave_info = fit_shape(means, mass / mass.sum(), times, True)
        error = max(float(np.max(np.abs(mono - pava(means, mass / mass.sum())))),
                    float(np.max(np.abs(concave - independent_slsqp(means, mass / mass.sum(), times)))))
        assert error <= 1e-7, (sequence, error)
        max_shape_error = max(max_shape_error, error)
        for name, values in (("free", means), ("increasing", mono), ("increasing_concave", concave)):
            fitted[name][index] = values[np.searchsorted(times, t[index])]
        negative_adjacent += int(np.sum(np.diff(means) < 0))
        total_adjacent += len(times) - 1
        slope = np.sum(mass * (times - np.average(times, weights=mass)) * means) / np.sum(mass * (times - np.average(times, weights=mass)) ** 2)
        negative_slopes += int(slope < 0)
        curve_rows.append({"peptide_sequence": sequence, "times": times.tolist(),
                           "replicate_means": means.tolist(), "time_weight": mass.tolist(),
                           "increasing": mono.tolist(), "increasing_concave": concave.tolist(),
                           "slope_per_minute": float(slope), "monotone_kkt": mono_info,
                           "concave_kkt": concave_info, "independent_prediction_error": error})
    pq.write_table(pa.Table.from_pylist([
        {**row, "target": float(y[j]), **{name: float(values[j]) for name, values in fitted.items()}}
        for j, row in enumerate(rows)]), ROOT / "results/shape-predictions.parquet")
    save("results/shape-curves.json", curve_rows)
    shape_mse = {name: float(w @ ((values - y) ** 2)) for name, values in fitted.items()}
    save("results/shape-summary.json", {"mse": shape_mse, "n_sequences": len(sequences),
         "n_observations": len(y), "negative_adjacent_mean_changes": negative_adjacent,
         "total_adjacent_mean_changes": total_adjacent, "negative_unpenalized_slopes": negative_slopes,
         "max_independent_solver_prediction_error": max_shape_error})

    fits = []
    for kind in ("uniform", "residue"):
        initial = model.initial(kind, len(amino_acids)).detach().numpy()
        low_rate = (1e-5 - 1e-7) / (1 - 1e-7)
        middle_tau = (60 - .001) / (120 - .001)
        low = np.array([math.log(low_rate / (1 - low_rate))] * (len(initial) - 1) +
                       [math.log(middle_tau / (1 - middle_tau))])
        for start_name, start in (("adam", np.array(original[kind]["raw_parameters"])),
                                  ("initial", initial), ("low_rate", low)):
            condition = f"{kind}-{start_name}"
            output = ROOT / "results" / condition
            output.mkdir()
            evaluations = []
            accepted = [start.copy()]
            began = time.monotonic()

            def objective(x):
                theta = torch.tensor(x, requires_grad=True)
                value, mse, penalty, _ = model.objective(theta, kind, train, aa_index)
                gradient = torch.autograd.grad(value, theta)[0].detach().numpy()
                evaluations.append(np.r_[time.monotonic() - began, value.item(), mse.item(), penalty.item(), x, gradient])
                return value.item(), gradient

            result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(-30., 30.)] * len(start),
                              callback=lambda x: accepted.append(x.copy()),
                              options={"maxiter": 1500, "maxfun": 3000, "maxls": 40,
                                       "gtol": 1e-9, "ftol": 1e-13})
            seconds = time.monotonic() - began
            theta = torch.tensor(result.x)
            prediction, offsets = model.predict(theta, kind, train, train, aa_index)
            _, tau, rates = model.decode(theta, kind, aa_index)
            value, mse, penalty, _ = model.objective(theta, kind, train, aa_index)
            np.savez_compressed(output / "optimizer-history.npz", evaluations=np.array(evaluations),
                                accepted=np.array(accepted), initial=start, final=result.x,
                                gradient=result.jac, offsets=offsets.numpy(),
                                evaluation_columns=np.array(["elapsed", "objective", "mse", "penalty"] +
                                    [f"theta_{i}" for i in range(len(start))] + [f"gradient_{i}" for i in range(len(start))]))
            pq.write_table(pa.Table.from_pylist([
                {**row, "target": float(y[j]), "prediction": float(prediction[j]), "condition": condition}
                for j, row in enumerate(rows)]), output / "predictions.parquet")
            summary = {"condition": condition, "kind": kind, "initialization": start_name,
                       "success": bool(result.success), "message": str(result.message), "iterations": int(result.nit),
                       "function_evaluations": int(result.nfev), "seconds": seconds,
                       "objective": value.item(), "mse": mse.item(), "penalty": penalty.item(),
                       "original_adam_objective": original[kind]["training_objective"],
                       "original_adam_mse": original[kind]["training_mse"],
                       "mse_improvement_percent": 100 * (1 - mse.item() / original[kind]["training_mse"]),
                       "tau_minutes": tau.item(), "rates_by_category": rates.tolist(),
                       "raw_parameters": result.x.tolist(), "raw_gradient": result.jac.tolist(),
                       "all_parameters_changed": bool(np.all(result.x != start)),
                       "gradient_absolute_totals": np.abs(np.array(evaluations)[:, 4 + len(start):]).sum(0).tolist(),
                       "raw_boundary_coordinates": np.flatnonzero(np.abs(result.x) >= 30 - 1e-7).tolist()}
            save(f"results/{condition}/summary.json", summary)
            fits.append(summary)
            print(json.dumps({k: summary[k] for k in ("condition", "success", "iterations", "seconds", "mse", "mse_improvement_percent", "tau_minutes")}), flush=True)
    decisions = {}
    for kind in ("uniform", "residue"):
        gain = max(r["mse_improvement_percent"] for r in fits if r["kind"] == kind)
        fraction = (shape_mse["increasing_concave"] - shape_mse["free"]) / original[kind]["training_mse"]
        decisions[kind] = {"optimizer_material_at_5pct": gain >= 5,
                           "best_training_mse_gain_percent": gain,
                           "shape_penalty_fraction_of_original_mse": fraction,
                           "shape_material_at_10pct": fraction >= .1}
    save("results/summary.json", {"scope": "training_only_no_heldout_evaluation", "shape_mse": shape_mse,
         "fits": fits, "decisions": decisions, "versions": {"numpy": np.__version__,
         "scipy": scipy.__version__, "torch": torch.__version__}, "biological_validation_claim": False})


if __name__ == "__main__":
    main()
