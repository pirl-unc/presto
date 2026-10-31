"""Frozen sparse bounded solver; unchanged training labels and independent-cut model."""

import gzip
import hashlib
import json
import math
import resource
import time

import numpy as np
from scipy.optimize import least_squares

from direct_rates import FAMILY, LN10, dump, load_blocks


def main():
    started = time.perf_counter()
    stage = json.loads((FAMILY / "reproduce/fitting.json").read_text())
    for name, expected in stage["source_hashes"].items():
        assert hashlib.sha256((FAMILY / name).read_bytes()).hexdigest() == expected
    audit_path = FAMILY / "results/audit.json"
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == stage["audit_sha256"]
    assert json.loads(audit_path.read_text())["qualification_passed"]
    _, blocks, saved = load_blocks()
    assert not (FAMILY / "results/direct-fits.json").exists()
    fits = []
    all_p, all_y = [], []
    mse = 0.0
    with gzip.open(FAMILY / "results/direct-predictions.jsonl.gz", "wt") as predictions:
        for number, block in enumerate(blocks):
            now = time.perf_counter()
            cache, history = {}, []
            prefix = f"condition-{number}"
            trace_path = FAMILY / f"results/{prefix}-evaluations.jsonl.gz"
            with gzip.open(trace_path, "wt") as trace:

                def evaluate(x):
                    if "x" not in cache or not np.array_equal(cache["x"], x):
                        residual, jac = block.residual_and_jacobian(x)
                        cache.update(x=x.copy(), residual=residual, jac=jac)
                        value = float(residual @ residual)
                        gradient = 2 * (jac.T @ residual)
                        row = dict(
                            evaluation=len(history) + 1,
                            mse=value,
                            gradient_max=float(np.max(np.abs(gradient))),
                            projected_gradient_mapping_max=float(
                                np.max(np.abs(x - np.maximum(0, x - gradient)))
                            ),
                            seconds=time.perf_counter() - now,
                        )
                        history.append(row)
                        trace.write(json.dumps(dict(**row, rates=x.tolist())) + "\n")
                        if len(history) % 25 == 0:
                            print(block.g["condition"], row, flush=True)
                    return cache["residual"], cache["jac"]

                def fun(x):
                    return evaluate(x)[0]

                def jac(x):
                    return evaluate(x)[1]

                def callback(x):
                    if time.perf_counter() - now > stage["max_seconds_per_condition"]:
                        raise StopIteration

                result = least_squares(
                    fun,
                    block.initial,
                    jac=jac,
                    bounds=(0, np.inf),
                    callback=callback,
                    **stage["solver"],
                )
            residual, final_jac = block.residual_and_jacobian(result.x)
            gradient = 2 * (final_jac.T @ residual)
            p = residual / block.sqrtw + block.g["y"]
            log_mass = block.reference_log_mass(result.x) / LN10
            value = float(residual @ residual)
            mse += value
            all_p.extend(p)
            all_y.extend(block.g["y"])
            mapping = result.x - np.maximum(0, result.x - gradient)
            record = dict(
                condition=block.g["condition"],
                parent=block.g["parent"],
                rates=result.x.tolist(),
                initial_rates=block.initial.tolist(),
                direct_mse_gradient=gradient.tolist(),
                active_mask=result.active_mask.tolist(),
                parameter_count=block.n,
                condition_weight=float(block.g["w"].sum()),
                initial_mse=history[0]["mse"],
                final_mse=value,
                condition_rmse=math.sqrt(value / block.g["w"].sum()),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                nfev=int(result.nfev),
                njev=int(result.njev),
                actual_computations=len(history),
                scipy_half_mse_optimality=float(result.optimality),
                projected_mse_gradient_mapping_max=float(np.max(np.abs(mapping))),
                direct_mse_gradient_max=float(np.max(np.abs(gradient))),
                complementarity_max=float(np.max(np.abs(result.x * gradient))),
                active_lower_bound_count=int(
                    np.count_nonzero(result.active_mask == -1)
                ),
                rate_below_1e_10_count=int(np.count_nonzero(result.x < 1e-10)),
                exact_zero_rate_count=int(np.count_nonzero(result.x == 0)),
                zero_reference_amount_rows=int(
                    np.count_nonzero(~np.isfinite(log_mass))
                ),
                reference_amount_below_1e_20_rows=int(np.count_nonzero(log_mass < -20)),
                minimum_finite_reference_log10_amount=float(
                    np.min(log_mass[np.isfinite(log_mass)])
                ),
                history=history,
                seconds=time.perf_counter() - now,
            )
            fits.append(record)
            dump(FAMILY / f"results/{prefix}-fit.json", record)
            for row, pred, weight, target, mass in zip(
                block.g["rows"], p, block.g["w"], block.g["y"], log_mass
            ):
                curve, t, ref, signal, reference_signal = row
                predictions.write(
                    json.dumps(
                        dict(
                            condition_id=block.g["condition"],
                            construct_id=block.g["parent"],
                            curve_id=curve["curve_id"],
                            split="train",
                            time=t,
                            reference_time=ref,
                            signal=signal,
                            reference_signal=reference_signal,
                            target=target,
                            weight=weight,
                            prediction=pred,
                            reference_log10_amount=float(mass)
                            if np.isfinite(mass)
                            else None,
                            zero_reference_amount=not np.isfinite(mass),
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        k: record[k]
                        for k in (
                            "condition",
                            "condition_rmse",
                            "nfev",
                            "status",
                            "projected_mse_gradient_mapping_max",
                            "active_lower_bound_count",
                            "seconds",
                        )
                    }
                ),
                flush=True,
            )
    from scipy.stats import pearsonr, spearmanr

    summary = dict(
        fits=fits,
        global_mse=mse,
        global_rmse=math.sqrt(mse),
        previous_mse=saved["free_condition_rates"]["final_mse"],
        pooled_pearson=float(pearsonr(all_p, all_y).statistic),
        pooled_spearman=float(spearmanr(all_p, all_y).statistic),
        prediction_sd=float(np.std(all_p)),
        target_sd=float(np.std(all_y)),
        observations=len(all_p),
        wall_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        interpretation="Training-only per-condition free-rate capacity diagnostic; no transferable sequence or concentration claim",
    )
    dump(FAMILY / "results/direct-fits.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "global_rmse",
                    "pooled_pearson",
                    "pooled_spearman",
                    "wall_seconds",
                    "peak_rss_bytes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
