"""Record a verified working checkpoint and its explicitly limited validation claim."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from report import table

E = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred", required=True)
    args = parser.parse_args()
    run = args.preferred
    assert run == "expanded-state-sqrt-s42-n6000", "primary seed, simpler shared-state recipe"
    fit = E / "results/fits" / run
    receipt = json.loads((fit / "receipt.json").read_text())
    checkpoint = fit / "raw/selected.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert receipt["coverage"] == "passed" and digest == receipt["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert not payload["frozen_parameters"] and not payload["frozen_targets"]
    assert payload["config"]["state_bottleneck"] == 64
    assert not payload["config"].get("elution_pathway_conditioning", False)
    checks = json.loads((E / "results/independent-verification.json").read_text())
    terminal = json.loads((E / "results/terminal-independent-verification.json").read_text())
    assert len(checks) == len(terminal) == 12
    assert run in {r["run"] for r in checks}
    metrics = pd.read_csv(E / "results/analysis/metrics.csv")
    controls = pd.read_csv(E / "results/context-ablation" / run / "analysis/metrics.csv")
    for _, control in controls[controls["mode"] == "original"].iterrows():
        reference = metrics[(metrics.model == run) & (metrics.population == control.population)
                            & (metrics["class"] == control["class"])].iloc[0]
        assert abs(control.auroc - reference.auroc) < 1e-12
    negative = pd.read_csv(E / "results/context-ablation/expanded-none-sqrt-s43-n6000/analysis/metrics.csv")
    assert negative.max_logit_change.max() < 1e-6
    guard = pd.read_csv(E / "results/analysis/affinity-retention.csv")
    affinity = guard[guard.model == run]
    assert len(affinity) == 4
    assert (affinity.delta_spearman >= -0.02).all() and (affinity.delta_rmse <= 0.03).all()
    comparison = []
    for pop, cls in [("expanded", "I"), ("expanded", "II"), ("apc", "I"),
                     ("apc", "II"), ("broad", "I"), ("matched", "I"), ("matched", "II")]:
        group = metrics[(metrics.population == pop) & (metrics["class"] == cls)]
        new = group[group.model == run].iloc[0]
        old = group[group.model == "previous-default"].iloc[0]
        if pop != "matched":
            assert new.auroc >= old.auroc
        cluster = "condition" if pop == "broad" else "protein"
        bootstrap = json.loads((E / f"results/analysis/paired-bootstrap-{run}-{cluster}.json").read_text())
        interval = next(r for r in bootstrap if r["population"] == pop and r["class"] == cls
                        and r["comparator"] == "previous-default" and r["metric"] == "auroc")
        comparison.append({
            "cohort": pop, "class": cls, "rows": int(new.n), "previous_AUROC": old.auroc,
            "shared_AUROC": new.auroc, "delta": new.auroc - old.auroc,
            "paired_95CI": f"[{interval['ci95'][0]:+.4f}, {interval['ci95'][1]:+.4f}]",
            "cluster": cluster, "previous_pair_win": old.pair_win, "shared_pair_win": new.pair_win,
        })
    decision = {
        "run": run, "local_checkpoint": str(checkpoint.relative_to(E)), "sha256": digest,
        "remote_checkpoint": f"{E.name}/fits/{run}/raw/selected.pt",
        "modal_volume": "presto-checkpoints", "core_commit": "5f9bfe37dfb5e82cecdf762ef1e59ff885668e40",
        "selected_step": payload["metadata"]["steps"], "selection": "equal-class/equal-study presentation BCE",
        "scope": "Working ordinary-presentation checkpoint for both classes; adaptive development validation",
        "architecture_choice": "Shared APC64, no optional pathway projection; no supported parameter frozen",
        "replication": "expanded-state-sqrt-s43-n6000", "test_accessed": False,
        "limitations": ["Affinity-matched class-I discrimination regresses versus previous default",
                        "Within-study APC shuffling does not establish isolated gene/drug effects",
                        "No quantitative expression or absolute presentation-yield training targets",
                        "Validation is reused for development; sealed TEST confirmation absent"],
        "comparison": comparison, "affinity_retention": affinity.to_dict("records"),
    }
    (E / "results/preferred-checkpoint.json").write_text(json.dumps(decision, indent=2) + "\n")
    pd.DataFrame(comparison).to_csv(E / "results/analysis/preferred-comparison.csv", index=False)
    text = [
        "# Working checkpoint decision", "",
        f"Use **{run}**, selected at update **{decision['selected_step']}** by the original "
        "equal-class/equal-study presentation BCE criterion. Both classes use the same "
        "end-to-end model, 64-dimensional APC state, centralized adapter and public predictor. "
        "The optional pathway projection is left off because its two-seed comparison gives "
        "no meaningful predictive advantage.", "",
        "All 12 fits and their selected/fixed-terminal comparisons are complete and independently "
        "verified. This is an ordinary-presentation working model, not a claim of a win on every "
        "processing benchmark or independent TEST confirmation.", "",
        table(pd.DataFrame(comparison)), "",
        "Intervals use 1,000 paired resamples and condition on these fitted models and development "
        "cohorts. They exclude model-selection and seed uncertainty. Sampled backgrounds are "
        "unobserved source windows, not measured biological nonbinders. Pooled AUROC and within-pair "
        "ordering answer different questions; both remain reported.", "",
        "## Measured affinity retention", "", table(affinity), "",
        "The original and ancestry-screened IC50 cohorts both pass the registered operational "
        "screen (no Spearman drop above .02 or log10(nM) RMSE rise above .03). Intrinsic affinity "
        "takes no APC input; its parameters were jointly trained, not frozen.", "",
        "## Attribution and remaining limits", "",
        "Broader native data and repaired study allocation account for most of the class-II "
        "improvement. Compare shared-state against no-state within each seed and at the same "
        "update endpoint to assess the APC increment. Older APC class-I gains replicate "
        "at 6,000 updates: +.00512 and +.00524 AUROC in seeds 42/43, with positive protein- "
        "and condition-cluster intervals. New class-II gains are +.01373 and +.01247. "
        "The primary seed's protein-cluster 95% interval is [.00027, .02760], but its "
        "condition-cluster interval crosses zero; both replication intervals cross zero. "
        "This is suggestive incremental class-II evidence, not a robustly resolved effect. "
        "Unknown/shuffled controls show predictive state associations, while little "
        "within-study signature variation limits intervention identification. Gene/drug "
        "effects and absolute KO severity remain unvalidated.", "",
        "The affinity-matched class-I regression is a real limitation of this default. "
        "Keep the previous checkpoint as an explicit matched-benchmark reference. The new model "
        "is chosen for ordinary presentation across both classes and retained measured affinity.", "",
        "## Reproduce and use", "",
        f"Checkpoint SHA256:`{digest}`. Local:`{decision['local_checkpoint']}`. "
        f"Remote:`presto-checkpoints:/{decision['remote_checkpoint']}`.", "",
        "Run `python code/predict_example.py` from this experiment directory using the shared "
        "Python environment; the helper validates the chosen checkpoint hash. "
        "See [USAGE.md](USAGE.md), [ARCHITECTURE.md](ARCHITECTURE.md), "
        "[data support](results/analysis/apc-training-support.md) and "
        "[complete comparison](results/analysis/report.md). TEST remains sealed.", "",
    ]
    (E / "DECISION.md").write_text("\n".join(text))
    print(json.dumps({k: decision[k] for k in ("run", "sha256", "selected_step", "local_checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
