"""Independent full-parent path/statistic replay and reference-gradient probes."""

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from path_reference import path_values

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(path.read_text())


def close(a, b, tolerance=1e-8):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert error <= tolerance, (error, tolerance)
    return error


def features(item):
    sequence = item["sequence"]
    x = np.zeros((len(sequence) - 1, 40))
    x[:, 0] = 1
    x[:, -1] = math.log(
        item["substrate_concentration_um"] / item["enzyme_substrate_denominator"] / 0.2
    )
    for i in range(len(sequence) - 1):
        left, right = sequence[i : i + 2]
        if left != "A":
            x[i, AA.index(left)] = 1
        if right != "A":
            x[i, 19 + AA.index(right)] = 1
    return x


def replay_path(state, path):
    logp, logq, sp, sq = path_values(state, path)
    error = max(
        close(logp, path["original_log_density"]),
        close(logq, path["proposal_log_density"]),
        close(logp - logq, path["log_weight"]),
        close(sp, path["original_score"]),
        close(sq, path["proposal_score"]),
    )
    return error, sp, sq


def main():
    tick = time.perf_counter()
    cfg = read(FAMILY / "reproduce/actual-inputs.json")
    inp = REPO / cfg["inputs_path"]
    assert hashlib.sha256(inp.read_bytes()).hexdigest() == cfg["inputs_sha256"]
    inputs = read(inp)
    summary = read(FAMILY / "results/actual-summary.json")
    assert summary["complete"]
    for key in ["previous_config", "previous_verification"]:
        source = REPO / cfg[key + "_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg[key + "_sha256"]
    previous_cfg = read(REPO / cfg["previous_config_path"])
    previous_verification = read(REPO / cfg["previous_verification_path"])
    assert previous_verification["all_replay_checks_pass"]
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
    assert len(summary["cases"]) == len(cfg["cases"])
    weights = np.array(cfg["log_readout_weights"])
    results = []
    ntraces = 0
    nraw = 0
    maxtrace = 0.0
    changed = 0
    for summary_row in summary["cases"]:
        caseid = summary_row["case_id"]
        case_path = FAMILY / f"results/actual-{caseid:02d}.json"
        case = read(case_path) if case_path.exists() else summary_row
        if not case_path.exists():
            assert case["status"] != "completed"
        if case["status"] != "completed":
            results.append(dict(case_id=caseid, status=case["status"]))
            continue
        item = inputs["cases"][case["input_index"]]
        x = features(item)
        logs = x @ np.array(inputs["initial_coefficients"])
        eta = case["eta"]
        size = len(logs) + 1
        targets = np.array(case["targets"])
        indices = (targets[:, 0], targets[:, 1] - 1)
        pin = cfg["reference_artifacts"][str(caseid)]
        for key in ["case", "array"]:
            source = REPO / pin[key + "_path"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest() == pin[key + "_sha256"]
            )
        previous = read(REPO / pin["case_path"])
        for key in [
            "case_id",
            "condition",
            "input_index",
            "eta",
            "n_residues",
            "targets",
        ]:
            assert case[key] == previous[key]
        archive = np.load(REPO / pin["array_path"])
        close(logs, archive["log_rates"])
        ref = case["reference"]
        for key in [
            "eligible",
            "positive_targets",
            "maximum_log_agreement",
            "log_probability",
            "gradient",
            "minimum_full_count_two",
            "minimum_full_count_four",
        ]:
            assert ref[key] == previous["reference"][key]
        checks = {
            "reference_reused_by_verified_hash": True,
            "independent_reference_recomputed": False,
        }
        if ref["eligible"]:
            for key in ["counts_two_steps", "counts_four_steps"]:
                values = archive[key][indices]
                assert (values > 0).all()
                close(np.log(values), ref["log_probability"], 1e-5)
            close(ref["gradient"], archive["reference_gradient"])
        if eta == 0:
            rates = np.exp(logs)
            lp = []
            for a, b in targets:
                value = -cfg["time_hours"] * rates[a : b - 1].sum()
                if a:
                    value += math.log(-math.expm1(-rates[a - 1] * cfg["time_hours"]))
                if b < size:
                    value += math.log(-math.expm1(-rates[b - 1] * cfg["time_hours"]))
                lp.append(value)
            checks["analytic_log_probability_error"] = close(lp, ref["log_probability"])
        close(
            -cfg["time_hours"] * np.exp(logs).sum() * (size / 20) ** eta,
            case["intact_log_probability"],
        )
        reviewed = []
        for row in case["rows"]:
            combined = []
            plain_combined = []
            logmeans = []
            for gi, group in enumerate(row["groups"]):
                values = []
                plain = []
                means = []
                for ti, g in enumerate(group["targets"]):
                    n = g["count"]
                    total = g["sum_scaled_weights"]
                    numerator = np.array(g["sum_scaled_weight_original_score"])
                    qnum = np.array(g["sum_scaled_weight_proposal_score"])
                    qsum = np.array(g["proposal_score_sum"])
                    controlled = (numerator - (total * qsum - qnum) / (n - 1)) / total
                    ordinary = numerator / total
                    lm = g["log_weight_max"] + math.log(total / n)
                    close(controlled, g["log_gradient"])
                    close(ordinary, g["plain_log_gradient"])
                    close(lm, g["log_mean"])
                    close(total * total / g["sum_scaled_weight_squared"], g["ess"])
                    values.append(controlled)
                    plain.append(ordinary)
                    means.append(lm)
                    if "raw_archive" in g:
                        raw = np.load(FAMILY / "results" / g["raw_archive"])
                        lw = raw["log_weight"]
                        scaled = np.exp(lw - lw.max())
                        close(lw, raw["logp"] - raw["logq"])
                        close(lw.max(), g["log_weight_max"])
                        close(scaled.sum(), total)
                        close(scaled @ raw["original_projected_score"], numerator, 1e-6)
                        close(scaled @ raw["proposal_projected_score"], qnum, 1e-6)
                        close(raw["proposal_projected_score"].sum(0), qsum, 1e-6)
                        assert (
                            int((np.exp(lw) == 0).sum()) == g["linear_underflow_count"]
                        )
                        nraw += 1
                        record = read(
                            FAMILY
                            / f"results/actual-{caseid:02d}-target-{ti}-traces.json"
                        )
                        close(record["state"]["rates"], np.exp(logs))
                        assert record["state"]["target"] == g["target"]
                        for pi, path in enumerate(record["paths"]):
                            error, sp, sq = replay_path(record["state"], path)
                            maxtrace = max(maxtrace, error)
                            ntraces += 1
                            close(
                                np.r_[sp[:-1] @ x, sp[-1]],
                                raw["original_projected_score"][pi],
                            )
                            close(
                                np.r_[sq[:-1] @ x, sq[-1]],
                                raw["proposal_projected_score"][pi],
                            )
                            close(path["log_weight"], lw[pi])
                combined.append(weights @ np.array(values))
                plain_combined.append(weights @ np.array(plain))
                logmeans.append(means)
                close(combined[-1], group["gradient"])
                close(plain_combined[-1], group["plain_gradient"])
            logmeans = np.array(logmeans)
            new = dict(
                count=row["count"],
                primary=row["primary"],
                original_worker_gate=row["gate_pass"],
                verified_gate=None,
            )
            if ref["log_probability"] is not None:
                lp = np.array(ref["log_probability"])
                relative = np.expm1(
                    logsumexp(logmeans, axis=0) - math.log(len(logmeans)) - lp
                )
                bias = (logmeans - lp).mean(0)
                mae = np.abs(logmeans - lp).mean(0)
                close(relative, row["mean_relative_probability_error"])
                close(bias, row["mean_log_probability_bias"])
                close(mae, row["mean_absolute_log_probability_error"])
                new.update(
                    mean_absolute_log_probability_error=mae.tolist(),
                    mean_log_probability_bias=bias.tolist(),
                    mean_relative_probability_error=relative.tolist(),
                )
            if ref["eligible"]:
                truth = np.array(ref["gradient"])
                norm = np.linalg.norm(truth)
                c = np.array(combined)
                p = np.array(plain_combined)
                me = float(np.linalg.norm(c.mean(0) - truth) / norm)
                rms = float(np.sqrt(((c - truth) ** 2).sum(1).mean()) / norm)
                close(me, row["mean_gradient_relative_error"])
                close(rms, row["rms_gradient_relative_error"])
                close(
                    np.linalg.norm(p.mean(0) - truth) / norm,
                    row["plain_mean_gradient_relative_error"],
                )
                close(
                    np.sqrt(((p - truth) ** 2).sum(1).mean()) / norm,
                    row["plain_rms_gradient_relative_error"],
                )
                gate = cfg["gates"]
                new.update(
                    mean_gradient_relative_error=me,
                    rms_gradient_relative_error=rms,
                    verified_gate=bool(
                        np.max(np.abs(relative))
                        <= gate["maximum_mean_relative_probability_error"]
                        and np.max(mae) <= gate["maximum_mean_abs_log_error"]
                        and me <= gate["maximum_mean_parameter_gradient_relative_error"]
                        and rms <= gate["maximum_rms_parameter_gradient_relative_error"]
                    ),
                )
                if new["verified_gate"] != row["gate_pass"]:
                    changed += 1
            reviewed.append(new)
        results.append(
            dict(
                case_id=caseid,
                status="verified",
                reference_eligible=ref["eligible"],
                checks=checks,
                rows=reviewed,
            )
        )
    assert changed == 0, "Worker and independent original-contract gate disagree"
    out = dict(
        all_replay_checks_pass=True,
        cases=results,
        n_raw_groups=nraw,
        n_traces=ntraces,
        maximum_trace_error=maxtrace,
        gate_decisions_changed_by_registered_mean_absolute_criterion=changed,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope="Reused previously independently checked exact references by hash, no new reference recomputation; all retained raw statistics and complete504 paths if all cases complete; unsaved individual paths not replayed; original mean-absolute gate verified",
    )
    (FAMILY / "results/actual-independent-verification.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
