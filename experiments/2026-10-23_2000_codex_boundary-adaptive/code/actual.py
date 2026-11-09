"""Bounded full-parent input-only importance and signed log-gradient study."""

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

from boundary import sample
from actual_metrics import probability_errors, probability_gate

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


class BudgetExceeded(RuntimeError):
    pass


def exceeded(signum, frame):
    raise BudgetExceeded("wall time" if signum == signal.SIGALRM else "resident memory")


def features(item):
    sequence = item["sequence"]
    x = np.zeros((len(sequence) - 1, 40))
    x[:, 0] = 1
    x[:, -1] = math.log(
        item["substrate_concentration_um"] / item["enzyme_substrate_denominator"] / 0.2
    )
    for i, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
        if left != "A":
            x[i, AA.index(left)] = 1
        if right != "A":
            x[i, 19 + AA.index(right)] = 1
    return x


def analytic_logs(logs, targets, t):
    rate = np.exp(logs)
    result = []
    for a, b in targets:
        value = -t * rate[a : b - 1].sum()
        if a:
            value += math.log(-math.expm1(-rate[a - 1] * t))
        if b < len(rate) + 1:
            value += math.log(-math.expm1(-rate[b - 1] * t))
        result.append(value)
    return np.array(result)


def reference(case, cfg, logs, x):
    started = time.perf_counter()
    source = cfg["reference_artifacts"][str(case["case_id"])]
    paths = {}
    for key in ["case", "array"]:
        path = REPO / source[key + "_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source[key + "_sha256"]
        paths[key] = path
    old = json.loads(paths["case"].read_text())
    for key in ["case_id", "condition", "input_index", "eta", "n_residues", "targets"]:
        assert old[key] == case[key]
    saved = np.load(paths["array"])
    np.testing.assert_allclose(saved["log_rates"], logs, atol=1e-12, rtol=0)
    ref = old["reference"].copy()
    if ref["eligible"]:
        np.testing.assert_allclose(
            saved["reference_gradient"], ref["gradient"], atol=1e-12, rtol=0
        )
    ref["previous_construction_seconds"] = ref["seconds"]
    ref["seconds"] = time.perf_counter() - started
    ref["reused_by_hash"] = True
    ref["reference_artifact"] = source
    return ref


def projected_group(result, x):
    p = np.column_stack(
        (
            result.original_score[:, :-1].numpy() @ x,
            result.original_score[:, -1].numpy(),
        )
    )
    q = np.column_stack(
        (
            result.proposal_score[:, :-1].numpy() @ x,
            result.proposal_score[:, -1].numpy(),
        )
    )
    lw = result.log_weight.numpy()
    maximum = float(lw.max())
    w = np.exp(lw - maximum)
    sw = float(w.sum())
    sp = w @ p
    sq = w @ q
    n = len(w)
    cv = sp - (sw * q.sum(0) - sq) / (n - 1)
    group = dict(
        log_weight_max=maximum,
        sum_scaled_weights=sw,
        sum_scaled_weight_squared=float(w @ w),
        sum_scaled_weight_original_score=sp.tolist(),
        sum_scaled_weight_proposal_score=sq.tolist(),
        proposal_score_sum=q.sum(0).tolist(),
        plain_log_gradient=(sp / sw).tolist(),
        log_gradient=(cv / sw).tolist(),
        log_mean=maximum + math.log(sw / n),
        ess=sw * sw / float(w @ w),
        linear_underflow_count=int((np.exp(lw) == 0).sum()),
    )
    return group, p, q


def worker(case, cfg, inputs):
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
    start = time.perf_counter()
    result = dict(**case, status="pending")
    conditional_sampling_seconds = 0.0
    try:
        item = inputs["cases"][case["input_index"]]
        x = features(item)
        logs = x @ np.array(inputs["initial_coefficients"])
        size = len(logs) + 1
        ref = reference(case, cfg, logs, x)
        result["reference"] = ref
        weights = np.array(cfg["log_readout_weights"])
        all_rows = []
        for ki, count in enumerate(cfg["counts"]):
            groups = []
            for gi in range(cfg["groups"]):
                target_groups = []
                for ti, (a, b) in enumerate(case["targets"]):
                    seed = (
                        cfg["seed_base"]
                        + 1000000 * case["case_id"]
                        + 10000 * ki
                        + 1000 * gi
                        + ti
                    )
                    sample_tick = time.perf_counter()
                    draws = sample(
                        torch.tensor(logs),
                        case["eta"],
                        a,
                        b,
                        cfg["time_hours"],
                        count,
                        generator=torch.Generator().manual_seed(seed),
                        record_paths=count == 512 and gi == 0,
                    )
                    conditional_sampling_seconds += time.perf_counter() - sample_tick
                    g, p, q = projected_group(draws, x)
                    g.update(seed=seed, target=[a, b], count=count)
                    if count == 512 and gi == 0:
                        raw = f"actual-{case['case_id']:02d}-target-{ti}-raw.npz"
                        np.savez_compressed(
                            FAMILY / "results" / raw,
                            log_weight=draws.log_weight.numpy(),
                            original_projected_score=p,
                            proposal_projected_score=q,
                            logp=draws.original_log_density.numpy(),
                            logq=draws.proposal_log_density.numpy(),
                            cuts=draws.cuts.numpy(),
                        )
                        g["raw_archive"] = raw
                        paths = []
                        for j in range(8):
                            paths.append(
                                dict(
                                    log_weight=float(draws.log_weight[j]),
                                    original_score=draws.original_score[j].tolist(),
                                    proposal_score=draws.proposal_score[j].tolist(),
                                    original_log_density=float(
                                        draws.original_log_density[j]
                                    ),
                                    proposal_log_density=float(
                                        draws.proposal_log_density[j]
                                    ),
                                    cuts=draws.cuts[j].tolist(),
                                    boundary_times=draws.boundary_times[j].tolist(),
                                    boundary_bonds=draws.boundary_bonds[j].tolist(),
                                    event_bonds=draws.event_bonds[j].tolist(),
                                    event_times=draws.event_times[j].tolist(),
                                    event_forced=draws.event_forced[j].tolist(),
                                )
                            )
                        (
                            FAMILY
                            / f"results/actual-{case['case_id']:02d}-target-{ti}-traces.json"
                        ).write_text(
                            json.dumps(
                                dict(
                                    state=dict(
                                        rates=np.exp(logs).tolist(),
                                        eta=case["eta"],
                                        time=cfg["time_hours"],
                                        target=[a, b],
                                    ),
                                    paths=paths,
                                ),
                                indent=2,
                            )
                            + "\n"
                        )
                    target_groups.append(g)
                combined = weights @ np.array(
                    [g["log_gradient"] for g in target_groups]
                )
                plain = weights @ np.array(
                    [g["plain_log_gradient"] for g in target_groups]
                )
                groups.append(
                    dict(
                        targets=target_groups,
                        gradient=combined.tolist(),
                        plain_gradient=plain.tolist(),
                    )
                )
            means = np.array([[t["log_mean"] for t in g["targets"]] for g in groups])
            gradients = np.array([g["gradient"] for g in groups])
            plain = np.array([g["plain_gradient"] for g in groups])
            diagnostics = {}
            if ref["log_probability"] is not None:
                lp = np.array(ref["log_probability"])
                rel, bias, mae = probability_errors(means, lp)
                diagnostics.update(
                    mean_relative_probability_error=rel.tolist(),
                    mean_log_probability_bias=bias.tolist(),
                    mean_absolute_log_probability_error=mae.tolist(),
                )
            gate = None
            if ref["eligible"]:
                truth = np.array(ref["gradient"])
                norm = np.linalg.norm(truth)
                me = float(np.linalg.norm(gradients.mean(0) - truth) / norm)
                rms = float(np.sqrt(((gradients - truth) ** 2).sum(1).mean()) / norm)
                diagnostics.update(
                    mean_gradient_relative_error=me,
                    rms_gradient_relative_error=rms,
                    plain_mean_gradient_relative_error=float(
                        np.linalg.norm(plain.mean(0) - truth) / norm
                    ),
                    plain_rms_gradient_relative_error=float(
                        np.sqrt(((plain - truth) ** 2).sum(1).mean()) / norm
                    ),
                )
                gates = cfg["gates"]
                gate = bool(
                    probability_gate(rel, mae, gates)
                    and me <= gates["maximum_mean_parameter_gradient_relative_error"]
                    and rms <= gates["maximum_rms_parameter_gradient_relative_error"]
                )
            all_rows.append(
                dict(
                    count=count,
                    primary=count == cfg["gates"]["primary_count"],
                    gate_pass=gate,
                    reference_eligible=ref["eligible"],
                    minimum_ess=min(t["ess"] for g in groups for t in g["targets"]),
                    mean_ess=float(
                        np.mean([t["ess"] for g in groups for t in g["targets"]])
                    ),
                    **diagnostics,
                    groups=groups,
                )
            )
        intact = sample(
            torch.tensor(logs),
            case["eta"],
            0,
            size,
            cfg["time_hours"],
            8,
            generator=torch.Generator().manual_seed(cfg["seed_base"] + case["case_id"]),
        )
        hazard = np.exp(logs) * (size / 20) ** case["eta"]
        truth = -cfg["time_hours"] * hazard.sum()
        score = np.r_[
            -cfg["time_hours"] * hazard,
            -cfg["time_hours"] * hazard.sum() * math.log(size / 20),
        ]
        result.update(
            status="completed",
            rows=all_rows,
            intact_log_probability=float(truth),
            intact_maximum_log_error=float(
                np.max(np.abs(intact.log_weight.numpy() - truth))
            ),
            intact_maximum_score_error=float(
                np.max(np.abs(intact.original_score.numpy() - score))
            ),
        )
    except BudgetExceeded as exc:
        result.update(status="budget_exceeded", error=str(exc))
    except Exception as exc:
        result.update(status="error", error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        done.set()
    result.update(
        seconds=time.perf_counter() - start,
        conditional_sampling_seconds=conditional_sampling_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / f"results/actual-{case['case_id']:02d}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["rows", "reference"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    cfg = json.loads((FAMILY / "reproduce/actual-inputs.json").read_text())
    path = REPO / cfg["inputs_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg["inputs_sha256"]
    inputs = json.loads(path.read_text())
    for key in ["previous_config", "previous_verification"]:
        source = REPO / cfg[key + "_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg[key + "_sha256"]
    previous_cfg = json.loads((REPO / cfg["previous_config_path"]).read_text())
    for key in [
        "cases",
        "time_hours",
        "counts",
        "groups",
        "seed_base",
        "seed_rule",
        "log_readout_weights",
        "gates",
    ]:
        assert cfg[key] == previous_cfg[key]
    assert json.loads((REPO / cfg["previous_verification_path"]).read_text())[
        "all_replay_checks_pass"
    ]
    frozen = json.loads((FAMILY / "reproduce/actual-execution.json").read_text())
    for path, digest in frozen["source_hashes"].items():
        assert hashlib.sha256((FAMILY / path).read_bytes()).hexdigest() == digest
    if args.worker is not None:
        worker(cfg["cases"][args.worker], cfg, inputs)
        return
    destination = FAMILY / "results/actual-summary.json"
    assert not destination.exists()
    tick = time.perf_counter()
    rows = []
    for case in cfg["cases"]:
        name = f"actual-{case['case_id']:02d}"
        print("START", name, flush=True)
        with (FAMILY / f"launch_logs/{name}.log").open("w") as out:
            try:
                p = subprocess.run(
                    [sys.executable, __file__, "--worker", str(case["case_id"])],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=cfg["runtime_limits"]["seconds"] + 15,
                )
                status = p.returncode
            except subprocess.TimeoutExpired:
                status = "hard_timeout"
        artifact = FAMILY / f"results/{name}.json"
        row = (
            json.loads(artifact.read_text())
            if artifact.exists()
            else dict(**case, status="worker_failed", returncode=status)
        )
        # Full group records remain in their per-case artifact.
        row = {k: v for k, v in row.items() if k != "rows"}
        if artifact.exists():
            row["primary"] = [
                {k: v for k, v in r.items() if k != "groups"}
                for r in json.loads(artifact.read_text()).get("rows", [])
                if r["primary"]
            ]
        rows.append(row)
        destination.write_text(
            json.dumps(dict(cases=rows, complete=False), indent=2) + "\n"
        )
        print("DONE", name, row["status"], row.get("seconds"), flush=True)
    destination.write_text(
        json.dumps(
            dict(cases=rows, complete=True, seconds=time.perf_counter() - tick),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
