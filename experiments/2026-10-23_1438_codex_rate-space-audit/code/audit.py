"""Qualify direct-rate mathematics and inspect saved training-only geometry."""

import csv
import hashlib
import json
import math
import platform
import resource
import time
from decimal import Decimal, localcontext

import numpy as np
import scipy
from scipy.optimize import least_squares

from direct_rates import FAMILY, LN10, boundary_ratio, dump, load_blocks


def quantiles(values):
    return dict(zip(("min", "p01", "p10", "median", "p90", "p99", "max"),
                    np.quantile(values, [0, .01, .1, .5, .9, .99, 1]).tolist()))


def spectrum(jac):
    gram = (jac.T @ jac).toarray()
    eigen = np.linalg.eigvalsh(gram)
    threshold = max(eigen[-1], 1e-300) * np.finfo(float).eps * len(eigen)
    positive = eigen > threshold
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    inv = 1 / np.maximum(norms, 1e-150)
    scaled = np.linalg.eigvalsh(gram * inv[:, None] * inv[None, :])
    return dict(eigenvalues=eigen.tolist(), numerical_rank=int(positive.sum()),
                rank_threshold=threshold, column_norms=quantiles(norms),
                resolved_condition_number=float(np.sqrt(eigen[-1] / eigen[positive][0])),
                column_scaled_eigenvalues=scaled.tolist(),
                warning="Local Gauss-Newton spectrum, not global identifiability; tiny eigenvalues limited by Gram roundoff")


def mathematical_checks(blocks):
    scalar = []
    with localcontext() as ctx:
        ctx.prec = 70
        for rate in (0., 1e-14, 1e-9, 1e-5, .000999, .001001, .01, 1., 100., 10000.):
            for t, ref in ((2., 1.), (24., 1.), (24., 4.)):
                r, td, rd = map(lambda v: Decimal(str(v)), (rate, t, ref))
                if r == 0:
                    v = (td / rd).ln()
                    d = -(td - rd) / 2
                else:
                    v = (1 - (-r * td).exp()).ln() - (1 - (-r * rd).exp()).ln()
                    d = td / ((r * td).exp() - 1) - rd / ((r * rd).exp() - 1)
                value, derivative = boundary_ratio(np.array([rate]), np.array([t]), np.array([ref]))
                scalar.append(dict(rate=rate, t=t, ref=ref,
                                   value_error=abs(float(value[0]) - float(v)),
                                   derivative_error=abs(float(derivative[0]) - float(d))))
    assert max(r["value_error"] for r in scalar) < 1e-12
    assert max(r["derivative_error"] for r in scalar) < 1e-9
    probes = []
    for block in blocks:
        rates = block.initial
        p, jac = block.prediction_and_jacobian(rates)
        g = block.g
        bond = np.zeros(len(g["sequence"]) - 1)
        valid = block.bond_index >= 0
        bond[valid] = rates[block.bond_index[valid]]
        old = np.array([-float(bond[a:b-1].sum()) * (t - ref)
                        for a, b, t, ref in zip(g["a"], g["b"], g["t"], g["ref"])])
        for row, (a, b, t, ref) in enumerate(zip(g["a"], g["b"], g["t"], g["ref"])):
            for index in ([a - 1] if a else []) + ([b - 1] if b < len(g["sequence"]) else []):
                old[row] += math.log(-math.expm1(-bond[index] * t)) - math.log(-math.expm1(-bond[index] * ref))
        positive_error = float(np.max(np.abs(p - old / LN10)))
        direction = np.sin(np.arange(block.n) + 1)
        anchor = np.full(block.n, .03)
        h = 1e-6
        plus = block.prediction_and_jacobian(anchor + h * direction)[0]
        minus = block.prediction_and_jacobian(anchor - h * direction)[0]
        expected = block.prediction_and_jacobian(anchor)[1] @ direction
        jvp_error = float(np.max(np.abs((plus - minus) / (2*h) - expected)))
        zero = np.zeros(block.n)
        direction = np.abs(direction)
        f0, j0 = block.prediction_and_jacobian(zero)
        f1 = block.prediction_and_jacobian(h * direction)[0]
        f2 = block.prediction_and_jacobian(2*h * direction)[0]
        zero_error = float(np.max(np.abs((-3*f0 + 4*f1 - f2)/(2*h) - j0 @ direction)))
        assert positive_error < 1e-9 and jvp_error < 1e-6 and zero_error < 1e-6
        probes.append(dict(condition=g["condition"], positive_prediction_error=positive_error,
                           finite_difference_jvp_error=jvp_error, zero_one_sided_jvp_error=zero_error))
    target = np.log(24.) + .1
    def fun(x):
        return boundary_ratio(x, np.array([24.]), np.array([1.]))[0] - target
    def jac(x):
        return boundary_ratio(x, np.array([24.]), np.array([1.]))[1][:, None]
    toy = least_squares(fun, [.1], jac=jac, bounds=(0, np.inf),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100)
    assert toy.x[0] < 1e-10 and abs(toy.fun[0] + .1) < 1e-9
    return dict(decimal_scalar_checks=scalar, block_checks=probes,
                constrained_toy=dict(rate=toy.x.tolist(), residual=toy.fun.tolist(),
                                     status=int(toy.status), nfev=int(toy.nfev), optimality=float(toy.optimality)))


def main():
    tick = time.perf_counter()
    cfg, blocks, saved = load_blocks()
    assert not (FAMILY / "results/audit.json").exists()
    stage = json.loads((FAMILY / "reproduce/audit.json").read_text())
    for name, sha in stage["source_hashes"].items():
        assert hashlib.sha256((FAMILY / name).read_bytes()).hexdigest() == sha
    qualification = mathematical_checks(blocks)
    dump(FAMILY / "results/qualification.json", qualification)
    audits, parameters, total = [], [], 0.
    for block in blocks:
        rates, g = block.initial, block.g
        now = time.perf_counter()
        residual, jac = block.residual_and_jacobian(rates)
        gradient = 2 * (jac.T @ residual)
        p = block.prediction_and_jacobian(rates)[0]
        log_mass = block.reference_log_mass(rates) / LN10
        response = np.log10([row[4] for row in g["rows"]]) - log_mass
        logjac = jac.multiply(rates[None, :]).tocsr()
        boundary_counts = np.zeros(block.n, dtype=int)
        for _, columns in block.boundaries:
            boundary_counts += np.bincount(columns, minlength=block.n)
        internal_counts = np.asarray((block.internal != 0).sum(axis=0)).ravel()
        norms = np.sqrt(np.asarray(jac.power(2).sum(axis=0)).ravel())
        for i, members in enumerate(block.members):
            parameters.append(dict(condition=g["condition"], local_group=i, global_group=block.offset+i,
                                   bonds=":".join(map(str, members)), rate_per_hour=rates[i],
                                   direct_mse_gradient=gradient[i], log_mse_gradient=gradient[i]*rates[i],
                                   weighted_jacobian_column_norm=norms[i],
                                   boundary_rows=boundary_counts[i], internal_rows=internal_counts[i]))
        mse = float(residual @ residual)
        total += mse
        record = dict(condition=g["condition"], parent=g["parent"], parameters=block.n,
                      observations=len(p), sparse_jacobian_nnz=jac.nnz,
                      weight=float(g["w"].sum()), contribution_to_global_mse=mse,
                      condition_rmse=math.sqrt(mse/g["w"].sum()), rate=quantiles(rates),
                      reference_log10_mass=quantiles(log_mass),
                      implied_log10_signal_per_copy=quantiles(response),
                      calibration_warning="Arbitrary source signal per expected fragment copy, not independently measured MS response or concentration",
                      boundary_groups=int((boundary_counts>0).sum()),
                      tiny_maximum_hazard_groups=int((rates*g["t"].max()<1e-6).sum()),
                      saturated_minimum_reference_boundary_groups=int(((boundary_counts>0)&(rates*g["ref"].min()>20)).sum()),
                      direct_gradient_max=float(np.max(np.abs(gradient))),
                      log_gradient_max=float(np.max(np.abs(gradient*rates))),
                      direct_spectrum=spectrum(jac), log_spectrum=spectrum(logjac),
                      build_and_spectrum_seconds=time.perf_counter()-now)
        audits.append(record)
        print(json.dumps({k:record[k] for k in ("condition","parameters","condition_rmse","direct_gradient_max","log_gradient_max")}),flush=True)
    assert abs(total-saved["free_condition_rates"]["final_mse"]) < 1e-10
    with (FAMILY / "results/parameter-audit.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(parameters[0]))
        writer.writeheader(); writer.writerows(parameters)
    result=dict(blocks=audits, global_mse=total, global_rmse=math.sqrt(total),
                previous_optimizer_history=saved["free_condition_rates"]["history"],
                previous_polish=saved["free_condition_rates"]["polish"],
                qualification_passed=True,wall_seconds=time.perf_counter()-tick,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__))
    dump(FAMILY / "results/audit.json",result)
    print(json.dumps({k:result[k] for k in ("global_rmse","qualification_passed","wall_seconds","peak_rss_bytes")}),flush=True)


if __name__ == "__main__":
    main()
