"""Complete-partition hybrid gradient variance; exact residuals and denominators."""

import json
import math
import resource
import time

import torch

from common import FAMILY, designs, dump, load_data, objective, predict, sha
from presto.models.independent_cut import (
    log_fragment_expectation,
    partition_contains,
    sample_partitions,
)


def prepare(theta, groups, design):
    prepared = []
    exact_data = torch.zeros_like(theta)
    for g, x in zip(groups, design):
        if g["split"] != "train":
            continue
        logs = x @ theta
        residual = predict(logs, g) - g["target"]
        coeff = 2 * g["weight"] * residual / math.log(10)
        spans, inverse = torch.stack((g["starts"], g["ends"]), dim=1).unique(
            dim=0, sorted=True, return_inverse=True
        )
        starts, ends = spans.T
        time_records = []
        for t in torch.cat((g["times"], g["refs"])).unique(sorted=True).tolist():
            combined = torch.zeros(len(spans), dtype=torch.float64)
            for values, sign in ((g["times"], 1), (g["refs"], -1)):
                mask = values == t
                combined.scatter_add_(0, inverse[mask], sign * coeff[mask])
            times = torch.full((len(spans),), t, dtype=torch.float64)
            logm = log_fragment_expectation(logs, starts, ends, times)
            m = logm.exp()
            assert (m > 0).all()
            h = logs.exp() * t
            cut_score = h * torch.exp(-h) / (-torch.expm1(-h))
            # Integrate irrelevant bonds analytically: E[score | fragment exists].
            internal = torch.zeros((len(spans), len(logs)), dtype=torch.float64)
            for row, (a, b) in enumerate(spans.tolist()):
                if a > 0:
                    internal[row, a - 1] = cut_score[a - 1]
                if b <= len(logs):
                    internal[row, b - 1] = cut_score[b - 1]
                internal[row, a : b - 1] = -h[a : b - 1]
            exact_data += combined @ (internal @ x)
            time_records.append(
                dict(
                    time=t,
                    log_rates=logs,
                    hazard=h,
                    cut_score=cut_score,
                    mass=m,
                    coefficient=combined,
                    starts=starts,
                    ends=ends,
                    design=x,
                    mean_event_probability=float(m.mean()),
                    minimum_event_probability=float(m.min()),
                )
            )
        prepared.append(
            dict(
                condition=g["condition"],
                records=time_records,
                inverse=inverse,
                times=g["times"],
                refs=g["refs"],
                prediction=predict(logs, g),
                weight=g["weight"],
            )
        )
    return prepared, exact_data


def estimate(prepared, count, generator):
    gradient = torch.zeros(
        prepared[0]["records"][0]["design"].shape[1], dtype=torch.float64
    )
    zero_events = 0
    events = 0
    partitions = 0
    bond_draws = 0
    presence_terms = 0
    undefined_weight = 0.0
    defined_weight = 0.0
    plugin_error = 0.0
    plugin_squared = 0.0
    for group in prepared:
        occurrence = {}
        for record in group["records"]:
            cuts, _ = sample_partitions(
                record["log_rates"], record["time"], count, generator=generator
            )
            present = partition_contains(cuts, record["starts"], record["ends"])
            occurrence[record["time"]] = present.sum(0)
            zero_events += int((occurrence[record["time"]] == 0).sum())
            events += len(record["mass"])
            score = (
                torch.where(cuts, record["cut_score"], -record["hazard"])
                @ record["design"]
            )
            reward = present.double() @ (record["coefficient"] / record["mass"])
            baseline = (reward.sum() - reward) / (count - 1)
            gradient += ((reward - baseline)[:, None] * score).mean(0)
            partitions += count
            bond_draws += cuts.numel()
            presence_terms += present.numel()
        for t, ref in (
            torch.stack((group["times"], group["refs"]), dim=1).unique(dim=0).tolist()
        ):
            mask = (group["times"] == t) & (group["refs"] == ref)
            index = group["inverse"][mask]
            a = occurrence[t][index]
            b = occurrence[ref][index]
            valid = (a > 0) & (b > 0)
            weights = group["weight"][mask]
            undefined_weight += float(weights[~valid].sum())
            defined_weight += float(weights[valid].sum())
            error = (
                torch.log10(a[valid].double() / b[valid])
                - group["prediction"][mask][valid]
            )
            plugin_error += float((weights[valid] * error).sum())
            plugin_squared += float((weights[valid] * error.square()).sum())
    return gradient, dict(
        partitions=partitions,
        bond_draws=bond_draws,
        presence_terms=presence_terms,
        zero_observed_span_time_fraction=zero_events / events,
        plugin_undefined_target_weight=undefined_weight,
        plugin_defined_target_weight=defined_weight,
        plugin_conditional_bias=None
        if defined_weight == 0
        else plugin_error / defined_weight,
        plugin_conditional_rmse=None
        if defined_weight == 0
        else math.sqrt(plugin_squared / defined_weight),
    )


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    config = json.loads((FAMILY / "reproduce/probe.json").read_text())
    assert sha(__file__) == config["sha256"]
    cfg, groups = load_data()
    fits = json.loads((FAMILY / "results/fits.json").read_text())
    design, basis, _ = designs(groups, "sequence_context")
    initial = torch.tensor(
        fits["sequence_context"]["checkpoints"]["0"], dtype=torch.float64
    )
    records = []
    reference_norm = None
    for state, values in [
        ("initial", initial),
        (
            "final",
            torch.tensor(
                fits["sequence_context"]["basis_coefficients"], dtype=torch.float64
            ),
        ),
    ]:
        theta = values.clone().requires_grad_()
        loss, _ = objective(theta, groups, design, basis, cfg["l2_nonintercept"])
        exact = torch.autograd.grad(loss, theta)[0]
        original = basis @ theta.detach()
        reg = original.clone() * cfg["l2_nonintercept"]
        reg[0] = 0
        reg = basis.T @ reg
        if reference_norm is None:
            reference_norm = float(exact.norm())
        prepared, integrated = prepare(theta.detach(), groups, design)
        integration_error = float((integrated + reg - exact).abs().max())
        assert integration_error < 1e-10
        for count in config["sample_counts"]:
            begin = time.monotonic()
            estimates = []
            stats = []
            for rep in range(config["estimates"]):
                generator = torch.Generator().manual_seed(
                    config["seed"]
                    + rep
                    + 1000 * count
                    + 1000000 * int(state == "final")
                )
                gradient, work = estimate(prepared, count, generator)
                estimates.append(gradient + reg)
                stats.append(work)
            values = torch.stack(estimates)
            mean = values.mean(0)
            se = values.std(0, unbiased=True) / math.sqrt(len(values))
            errors = values - exact
            absolute_rms = float(errors.square().sum(1).mean().sqrt())
            record = dict(
                state=state,
                count=count,
                exact_gradient=exact.tolist(),
                gradients=values.tolist(),
                mean_gradient=mean.tolist(),
                standard_errors=se.tolist(),
                integrated_conditional_gradient_error=integration_error,
                exact_gradient_norm=float(exact.norm()),
                mean_error_norm=float((mean - exact).norm()),
                rms_error_norm=absolute_rms,
                rms_relative_to_initial_gradient=absolute_rms / reference_norm,
                rms_relative_to_current_gradient=absolute_rms
                / max(float(exact.norm()), 1e-30),
                max_coordinate_mean_error_in_standard_errors=float(
                    ((mean - exact).abs() / se.clamp_min(1e-30)).max()
                ),
                work=stats,
                wall_seconds=time.monotonic() - begin,
                mean_plugin_undefined_weight=sum(
                    w["plugin_undefined_target_weight"] for w in stats
                )
                / len(stats),
                event_probability_range=[
                    min(
                        r["minimum_event_probability"]
                        for g in prepared
                        for r in g["records"]
                    ),
                    max(float(r["mass"].max()) for g in prepared for r in g["records"]),
                ],
            )
            records.append(record)
            dump(FAMILY / "results/gradient-probes.json", records)
            print(
                state,
                count,
                "RMS/initial",
                record["rms_relative_to_initial_gradient"],
                "undefined plug-in",
                record["mean_plugin_undefined_weight"],
                "seconds",
                record["wall_seconds"],
                flush=True,
            )
    dump(
        FAMILY / "results/probe-runtime.json",
        dict(
            wall_seconds=time.monotonic() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            exact_objective_denominator_hybrid=True,
            iterative_sampled_fit=False,
            zero_count_imputation=False,
        ),
    )


if __name__ == "__main__":
    main()
