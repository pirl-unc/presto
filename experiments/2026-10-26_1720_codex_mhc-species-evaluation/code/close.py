"""Close saved evidence into the experiment narrative and canonical registry."""

from __future__ import annotations

import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
START = "<!-- mhc-species-evaluation-20261026:start -->"
END = "<!-- mhc-species-evaluation-20261026:end -->"


def read(path):
    return json.loads(path.read_text())


def replace_section(path, section):
    text = path.read_text()
    if START in text:
        before, remainder = text.split(START, 1)
        _, after = remainder.split(END, 1)
        text = before + section + after
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    temporary = path.with_suffix(".pending")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    result = read(FAMILY / "results/comparison.json")
    launch = read(FAMILY / "reproduce/launch.json")
    baseline = FAMILY / "results/baselines/independent-verification.json"
    if not baseline.exists() or read(baseline)["status"] != "passed":
        raise RuntimeError("baseline source/numeric replay has not passed")
    complete = result["status"] == "completed" and all(
        read(FAMILY / "results" / c / "independent-verification.json")["status"] == "passed"
        for c in ("seed42", "seed43", "seed44")
    )
    rows = []
    for condition, splits in result["metrics"].items():
        train, val, test = (splits[s] for s in ("train", "validation", "test"))
        rows.append(
            f"| {condition} | {train['ce']:.6f} | {val['ce']:.6f} | "
            f"{val['balanced_accuracy']:.6f} | {val['component_mean_ce']:.6f} | "
            f"{test['ce']:.6f} | {test['balanced_accuracy']:.6f} | "
            f"{test['component_mean_ce']:.6f} |"
        )
    state_path = FAMILY / "reproduce/continuation-state.json"
    state = read(state_path) if state_path.exists() else {}
    status = (
        "Completed; narrow engineering screen **" + result["engineering_screen"] + "**."
        if complete
        else "Incomplete; fixed controls independently verified; neural conditions unfinished."
    )
    if state.get("status") == "failed":
        status += " Continuation stopped: " + str(state.get("failure")) + "."
    receipts = sorted((FAMILY / "reproduce").glob("baselines-*.json"))
    runtime = read(receipts[-1]) if receipts else {}
    lines = [
        START,
        "## MHC reference species evaluation — 2026-10-26_1720_codex",
        "",
        "Date: 2026-10-26. Agent/model: Codex / GPT-6.",
        "",
        status,
        "",
        "Experiment directory: `experiments/2026-10-26_1720_codex_mhc-species-evaluation/`.",
        "Exact contract and reproducibility bundle: `reproduce/plan.md`, `reproduce/launch.json`,",
        "`reproduce/launch.sh`, `reproduce/verify-v2.sh` and frozen `reproduce/source/`.",
        "Initial executed commit: `"
        + launch["git_commit"]
        + "`; exact dirty state is in the receipt.",
        "Original data/runtime remain unchanged; generated audit files were untracked.",
        "",
        "Data: 13,218 four-species single-domain reference annotations (6,092/694/6,432",
        "train/validation/test), original aliases and fixed exact-domain/reference-chain",
        "components. Human/mouse/rat/rhesus macaque map to the shared mhc_species head",
        "through their actually observed mhc1 or mhc2 input. No pairing/class labels,",
        "synthetic data, assays, bounds or censoring. Regression/500 nM metrics do not apply.",
        "No affinity pretraining/transfer is evaluated. Prior and length/composition/slot",
        "softmax controls fit only training data; softmax uses summed CE plus L2 alpha1,",
        "unpenalized intercept and training-only standardization. Neural seeds42/43/44",
        "use d128/two layers/four heads, dropout0.1, AdamW3e-4, decay1e-4, CE weight1,",
        "batch64, max50/min10 epochs, patience10, minimum validation CE selection and",
        "one selected-checkpoint test evaluation. Neural conditions run after IC50 serially.",
        "",
        "Local CPU only; no GPU/Modal requested or observed. Existing Python3.12.6;",
        "read-only pinned dependency overlay, one control/verifier thread or four neural",
        "threads, interop one. Exact environment and hardware are in each receipt.",
        f"Fixed-control elapsed time: {runtime.get('elapsed_seconds', 0):.3f}s; "
        f"peak process RSS: {runtime.get('process_peak_rss_bytes', 'unavailable')} bytes.",
        "",
        (
            "| Condition | Train CE | Validation CE | Validation balanced accuracy | "
            "Validation component CE | Test CE | Test balanced accuracy | Test component CE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Full per-species/slot discrimination, confusion, Brier, calibration, probability",
        "variation and component metrics are in `results/comparison.json`; the concise",
        "table is `results/comparison.csv`. Every split's predictions are preserved in",
        "`results/baselines/*-predictions.jsonl.gz` and, as completed, each neural result",
        "directory. Raw checkpoints and native dumps: `" + launch["raw_artifacts"] + "`.",
        "",
        "Independent verification checks all original source fields plus precisely the",
        "manifest evidence-policy normalization, all metrics and control probabilities,",
        "training-only feature scaling and the analytic convex-objective gradient.",
        "The original raw-row equality checker rejected that known loader-added field;",
        "its source, failed receipt and log are retained. The v2 correction and exact",
        "source/prediction hashes are in `reproduce/verification-amendment.md` and",
        "`reproduce/verification-v2-contract.json`. No control was refit and no test rescored.",
        "",
        "Interpretation: the composition control improves validation discrimination over",
        "the prior but remains below the prespecified neural screen. Test observation CE",
        "favors the prior while equal-component CE favors composition: the existing",
        "5,664-row human component dominates the observation-weighted result. Neither",
        "statistic establishes homology-independent or population-level generalization.",
        "There is no qualified neural winner until all three fixed seeds pass. Even a",
        "passing species screen would leave affinity transfer, class/pairing inference,",
        "APC/MS/TCR science and the PR merge criteria open.",
        "",
        "Status: " + ("closed" if complete else "active; not scientifically complete") + ".",
        "Next step: "
        + (
            "review all seed screens and design matched transfer separately."
            if complete
            else "finish serial neural fits, independent replay and closure."
        ),
        "Open questions: reference generalization and useful affinity transfer.",
        END,
    ]
    section = "\n".join(lines)
    replace_section(FAMILY / "README.md", section)
    replace_section(ROOT / "experiments/experiment_log.md", section)
    print(status, flush=True)


if __name__ == "__main__":
    main()
