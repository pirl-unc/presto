"""Fresh-process CRAM runtime on the identical frozen actual-parent inputs."""

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

from rational import expectations

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


class BudgetExceeded(RuntimeError):
    pass


def exceeded(signum, frame):
    raise BudgetExceeded("wall time" if signum == signal.SIGALRM else "resident memory")


def inputs():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    prior = REPO / cfg["prior_family"]
    data = prior / "results/runtime-inputs.json"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == cfg["prior_inputs_sha256"]
    return cfg, prior, json.loads(data.read_text())


def rates(item, coefficients):
    seq = item["sequence"]
    x = np.zeros((len(seq) - 1, 40))
    x[:, 0] = 1
    x[:, -1] = math.log(
        item["substrate_concentration_um"] / item["enzyme_substrate_denominator"] / 0.2
    )
    for i, (left, right) in enumerate(zip(seq[:-1], seq[1:])):
        if left != "A":
            x[i, AA.index(left)] = 1
        if right != "A":
            x[i, 19 + AA.index(right)] = 1
    return x @ np.array(coefficients)


def worker(case, cfg, data):
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM, exceeded)
    signal.signal(signal.SIGUSR1, exceeded)
    signal.setitimer(signal.ITIMER_REAL, cfg["runtime_limits"]["seconds"])
    done = threading.Event()

    def watch():
        while not done.wait(0.1):
            if (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                > cfg["runtime_limits"]["rss_bytes"]
            ):
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    threading.Thread(target=watch, daemon=True).start()
    tick = time.perf_counter()
    result = dict(**case, status="pending")
    try:
        item = data["cases"][case["input_index"]]
        logs = torch.tensor(
            rates(item, data["initial_coefficients"]),
            dtype=torch.float64,
            requires_grad=True,
        )
        eta = torch.tensor(case["eta"], dtype=torch.float64, requires_grad=True)
        size = len(logs) + 1
        index = torch.arange(size, dtype=torch.float64)
        weight = (
            torch.sin(index[:, None] + 1) * torch.cos(index[None, :] + 0.3) + 1.1
        ) / size
        start = time.perf_counter()
        counts = expectations(
            logs, eta, 24, order=case["order"], substeps=case["substeps"]
        )
        forward = time.perf_counter() - start
        scalar = (counts * weight).sum()
        gradient = torch.autograd.grad(scalar, (logs, eta))
        backward = time.perf_counter() - start - forward
        arr = counts.detach().numpy()
        grad = np.r_[gradient[0].numpy(), float(gradient[1])]
        length = np.arange(size)[None, :] - np.arange(size)[:, None] + 1
        valid = length > 0
        assert np.isfinite(arr).all() and np.isfinite(grad).all()
        np.savez_compressed(
            FAMILY / f"results/{case['name']}.npz",
            counts=arr,
            gradient=grad,
            log_rates=logs.detach().numpy(),
        )
        result.update(
            status="completed",
            forward_seconds=forward,
            backward_seconds=backward,
            readout=float(scalar.detach()),
            mass_error=abs(float((arr * length * valid).sum()) - size),
            negative_count=int((arr[valid] < 0).sum()),
            minimum_count=float(arr[valid].min()),
            log_admission_gate=bool((arr[valid] >= 0).all()),
        )
    except BudgetExceeded as exc:
        result.update(status="budget_exceeded", error=str(exc))
    except Exception as exc:
        result.update(status="error", error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        done.set()
    result.update(
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / f"results/{case['name']}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--worker", type=int)
    args = args.parse_args()
    cfg, prior, data = inputs()
    frozen = json.loads((FAMILY / "reproduce/runtime.json").read_text())
    for path, digest in frozen["source_hashes"].items():
        assert hashlib.sha256((FAMILY / path).read_bytes()).hexdigest() == digest
    cases = frozen["cases"]
    if args.worker is not None:
        worker(cases[args.worker], cfg, data)
        return
    tick = time.perf_counter()
    records = []
    summary = FAMILY / "results/runtime-summary.json"
    assert not summary.exists(), "Refuse to overwrite an executed runtime suite"
    for i, case in enumerate(cases):
        print("START", case["name"], flush=True)
        with (FAMILY / f"launch_logs/{case['name']}.log").open("w") as stream:
            try:
                process = subprocess.run(
                    [sys.executable, __file__, "--worker", str(i)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=cfg["runtime_limits"]["seconds"] + 15,
                )
                status = process.returncode
            except subprocess.TimeoutExpired:
                status = "hard_timeout"
        path = FAMILY / f"results/{case['name']}.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else dict(**case, status="worker_failed", returncode=status)
        )
        records.append(result)
        summary.write_text(
            json.dumps(dict(cases=records, complete=False), indent=2) + "\n"
        )
        print("DONE", json.dumps(result), flush=True)
    summary.write_text(
        json.dumps(
            dict(cases=records, complete=True, seconds=time.perf_counter() - tick),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
