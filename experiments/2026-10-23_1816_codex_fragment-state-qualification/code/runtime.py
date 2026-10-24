"""Input-only actual-parent CPU/precision comparison with explicit time/RSS gates."""

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from presto.models.fragment_state import fragment_expectations, sample_fragment_paths
from presto.models.independent_cut import (
    log_fragment_expectation,
    ordered_bond_features,
)

FAMILY = Path(__file__).resolve().parents[1]


class BudgetExceeded(RuntimeError):
    pass


def budget_signal(signum, frame):
    raise BudgetExceeded(
        "wall-time budget" if signum == signal.SIGALRM else "resident-memory budget"
    )


def run_case(case, backend, cfg, inputs):
    theta = torch.tensor(inputs["initial_coefficients"], dtype=torch.float64)
    context = math.log(
        case["substrate_concentration_um"] / case["enzyme_substrate_denominator"] / 0.2
    )
    x = ordered_bond_features(case["sequence"], context=context)
    logs = (x @ theta).requires_grad_()
    eta = torch.tensor(case["eta"], dtype=torch.float64, requires_grad=True)
    size = len(logs) + 1
    i = torch.arange(size, dtype=torch.float64)
    weight = (torch.sin(i[:, None] + 1) * torch.cos(i[None, :] + 0.3) + 1.1) / size
    last, first = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy")
    lengths = last - first + 1
    valid = lengths > 0
    prefix = f"runtime-{case['case_id']:02d}-{backend}"
    tick = time.perf_counter()
    if backend == "exact":
        result = fragment_expectations(
            logs, eta, [cfg["time_hours"]], absolute_error=cfg["absolute_error"]
        )
        counts = result.counts[0]
        forward = time.perf_counter() - tick
        scalar = (counts * weight).sum()
        grads = torch.autograd.grad(scalar, (logs, eta))
        backward = time.perf_counter() - tick - forward
        mass = float((counts.detach() * lengths * valid).sum())
        assert abs(mass - size) < 1e-8 and (counts >= 0).all()
        reference_error = reference_gradient_error = None
        if case["eta"] == 0:
            starts, ends = torch.triu_indices(size, size)
            ends = ends + 1
            analytic = log_fragment_expectation(
                logs,
                starts,
                ends,
                torch.full(starts.shape, cfg["time_hours"], dtype=torch.float64),
            ).exp()
            reference_error = float(
                (analytic - counts.detach()[starts, ends - 1]).abs().max()
            )
            analytical_gradient = torch.autograd.grad(
                analytic @ weight[starts, ends - 1], logs
            )[0]
            reference_gradient_error = float(
                (analytical_gradient - grads[0]).abs().max()
            )
            assert reference_error <= 1e-10 and reference_gradient_error <= 2e-7
        np.savez_compressed(
            FAMILY / f"results/{prefix}.npz",
            counts=counts.detach().numpy(),
            log_rate_gradient=grads[0].numpy(),
            eta_gradient=grads[1].numpy(),
        )
        metrics = dict(
            forward_seconds=forward,
            backward_seconds=backward,
            readout=float(scalar.detach()),
            mass=mass,
            mass_error=abs(mass - size),
            minimum_count=float(counts.detach().min()),
            poisson_steps=result.poisson_steps,
            operator_applications=result.operator_applications,
            error_bound=result.expected_count_error_bound,
            independent_cut_amount_error=reference_error,
            independent_cut_gradient_error=reference_gradient_error,
            log_rate_gradient_norm=float(grads[0].norm()),
            eta_gradient=float(grads[1]),
        )
    else:
        samples = sample_fragment_paths(
            logs.detach(),
            eta.detach(),
            cfg["time_hours"],
            cfg["sample_count"],
            generator=torch.Generator().manual_seed(cfg["seed"] + case["case_id"]),
        )
        sample_seconds = time.perf_counter() - tick
        cuts = samples.cuts.numpy()
        rewards = []
        counts = np.zeros((size, size))
        mass_error = 0
        w = weight.numpy()
        for cut in cuts:
            bounds = [0, *list(np.flatnonzero(cut) + 1), size]
            mass_error = max(
                mass_error,
                abs(sum(b - a for a, b in zip(bounds[:-1], bounds[1:])) - size),
            )
            rewards.append(sum(w[a, b - 1] for a, b in zip(bounds[:-1], bounds[1:])))
            for a, b in zip(bounds[:-1], bounds[1:]):
                counts[a, b - 1] += 1 / len(cuts)
        score = np.column_stack(
            (samples.log_rate_score.numpy(), samples.length_score.numpy())
        )
        rewards = np.array(rewards)
        gradient = (
            len(cuts)
            / (len(cuts) - 1)
            * ((rewards[:, None] * score).mean(0) - rewards.mean() * score.mean(0))
        )
        assert mass_error == 0 and np.isfinite(gradient).all()
        np.savez_compressed(
            FAMILY / f"results/{prefix}.npz",
            cuts=cuts,
            path_scores=score,
            path_rewards=rewards,
            mean_counts=counts,
            loo_gradient=gradient,
        )
        metrics = dict(
            sample_seconds=sample_seconds,
            readout=float(rewards.mean()),
            sample_count=len(cuts),
            mass_error=mass_error,
            mean_events=float(samples.event_counts.double().mean()),
            log_rate_gradient_norm=float(np.linalg.norm(gradient[:-1])),
            eta_gradient=float(gradient[-1]),
            scope="One input-only K512 group; a runtime/precision diagnostic, not a biological gradient qualification gate",
        )
    return metrics


def worker(case_id, backend, cfg):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = json.loads((FAMILY / "results/runtime-inputs.json").read_text())
    case = cfg["cases"][case_id]
    source = inputs["cases"][case["input_index"]]
    case = {**source, **case}
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.1):
            if (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                > cfg["max_rss_bytes"]
            ):
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    signal.signal(signal.SIGALRM, budget_signal)
    signal.signal(signal.SIGUSR1, budget_signal)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    signal.setitimer(signal.ITIMER_REAL, cfg["max_seconds"])
    tick = time.perf_counter()
    metrics = {}
    try:
        metrics = run_case(case, backend, cfg, inputs)
        status = "passed"
        error = None
    except BudgetExceeded as exc:
        status = "budget_exceeded"
        error = str(exc)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
    record = dict(
        case_id=case_id,
        condition=case["condition"],
        n_residues=case["n_residues"],
        eta=case["eta"],
        backend=backend,
        status=status,
        error=error,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metrics=metrics,
    )
    path = FAMILY / f"results/runtime-{case_id:02d}-{backend}.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int)
    parser.add_argument("--backend", choices=["exact", "sampled"])
    args = parser.parse_args()
    cfg = json.loads((FAMILY / "reproduce/runtime.json").read_text())
    for name, digest in cfg["source_hashes"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256(
            (FAMILY / "results/runtime-inputs.json").read_bytes()
        ).hexdigest()
        == cfg["inputs_sha256"]
    )
    if args.case is not None:
        return worker(args.case, args.backend, cfg)
    records = []
    tick = time.perf_counter()
    for case in cfg["cases"]:
        for backend in ["exact", "sampled"]:
            name = f"runtime-{case['case_id']:02d}-{backend}"
            path = FAMILY / f"results/{name}.json"
            assert not path.exists()
            print("START", name, case, flush=True)
            with (FAMILY / f"launch_logs/{name}.log").open("w") as stream:
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--case",
                            str(case["case_id"]),
                            "--backend",
                            backend,
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=cfg["max_seconds"] + 15,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    returncode = "hard_timeout"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                record = dict(
                    case_id=case["case_id"],
                    backend=backend,
                    status="worker_failed",
                    returncode=returncode,
                )
            records.append(record)
            (FAMILY / "results/runtime-summary.json").write_text(
                json.dumps(
                    dict(
                        cases=records,
                        seconds=time.perf_counter() - tick,
                        complete=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print("DONE", json.dumps(record), flush=True)
    (FAMILY / "results/runtime-summary.json").write_text(
        json.dumps(
            dict(
                cases=records,
                seconds=time.perf_counter() - tick,
                complete=True,
                all_exact_cases_pass=all(
                    r["status"] == "passed" for r in records if r["backend"] == "exact"
                ),
                all_sampled_cases_pass=all(
                    r["status"] == "passed"
                    for r in records
                    if r["backend"] == "sampled"
                ),
                source_targets_loaded=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
