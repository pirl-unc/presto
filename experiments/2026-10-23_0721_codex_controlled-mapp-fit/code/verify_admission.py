"""Independent source/run replay of the shared contrast, no Hitlist imports."""

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
started = time.perf_counter()
base, contrast = Path(CFG["run_recipe"]), Path(CFG["contrast_output"])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


assert digest(base / "manifest.json") == CFG["run_recipe_manifest_sha256"]
assert (
    digest(Path(CFG["source_package"]) / "manifest.json")
    == CFG["source_manifest_sha256"]
)
manifest = json.loads((contrast / "manifest.json").read_text())
assert manifest["source_package_manifest_sha256"] == CFG["source_manifest_sha256"]
for name, item in manifest["files"].items():
    assert digest(contrast / name) == item["sha256"]
for old, new in (
    ("source_tables.json", "source_tables.json"),
    ("examples.json", "run_examples.json"),
    ("dispositions.json", "run_dispositions.json"),
    ("components.json", "components.json"),
):
    assert digest(base / old) == digest(contrast / new)
rows = json.loads((base / "examples.json").read_text())
source_peptides = {
    r["peptide_row_id"]: r for r in json.loads((base / "components.json").read_text())
}
groups = defaultdict(dict)
for row in rows:
    if (
        row["source_dataset"] == "PXD009508"
        and row["context"]["capture_label_raw"] == "alpha6"
    ):
        key = (row["context"]["treatment_label_raw"], row["context"]["replicate_label"])
        assert key not in groups[row["peptide_row_id"]]
        groups[row["peptide_row_id"]][key] = row
examples = {
    r["peptide_row_id"]: r for r in json.loads((contrast / "examples.json").read_text())
}
dispositions = {
    r["peptide_row_id"]: r
    for r in json.loads((contrast / "dispositions.json").read_text())
}
assert dispositions.keys() == source_peptides.keys()
admitted, reason_counts = set(), Counter()
for pid, component in source_peptides.items():
    reasons = set()
    if not pid.startswith("pxd009508:"):
        reasons.add("outside_inhibitor_project")
    else:
        for arm in ("ut", "epoxo"):
            arm_rows = [
                r for (treatment, _), r in groups[pid].items() if treatment == arm
            ]
            if len(arm_rows) != 3:
                reasons.add("fewer_than_three_positive_runs_" + arm)
            if (
                sum(r["evidence_counts"]["n_direct_msms_evidence"] for r in arm_rows)
                < 1
            ):
                reasons.add("no_direct_msms_" + arm)
    assert set(dispositions[pid]["reasons"]) == reasons
    assert dispositions[pid]["status"] == ("excluded" if reasons else "admitted")
    reason_counts.update(reasons)
    if reasons:
        continue
    admitted.add(pid)
    out = examples[pid]
    for key, value in component.items():
        assert out[key] == value
    assert out["peptide_sequence"] == groups[pid][("ut", 1)]["peptide_sequence"]
    for arm, label in (("ut", "untreated"), ("epoxo", "epoxomicin")):
        values = [groups[pid][(arm, i)]["reported_intensity"] for i in (1, 2, 3)]
        ids = [groups[pid][(arm, i)]["measurement_id"] for i in (1, 2, 3)]
        assert (
            out[label + "_intensities"] == values
            and out[label + "_measurement_ids"] == ids
        )
        assert out[label + "_mean_intensity"] == math.fsum(values) / 3
assert set(examples) == admitted
summary = json.loads((contrast / "summary.json").read_text())
assert summary["reason_counts"] == reason_counts
assert summary["examples_by_split"] == {"train": 59, "val": 14, "test": 14}
prior = json.loads(
    (
        ROOT.parent
        / "2026-10-23_0645_codex_controlled-mapp-signal/results/training-complete-case-values.json"
    ).read_text()
)
assert {r["peptide_row_id"] for r in prior} == {
    r["peptide_row_id"] for r in examples.values() if r["split"] == "train"
}
for r in prior:
    assert examples[r["peptide_row_id"]]["untreated_intensities"] == r["ut"]
    assert examples[r["peptide_row_id"]]["epoxomicin_intensities"] == r["epoxo"]
result = {
    "passed": True,
    "n_source_peptide_decisions": len(dispositions),
    "n_admitted_peptides": len(admitted),
    "n_original_run_values": 6 * len(admitted),
    "examples_by_split": summary["examples_by_split"],
    "old_run_recipe_payloads_unchanged": True,
    "prior_training_signal_cohort_identical": True,
    "held_out_policy": "Source arithmetic and frozen eligibility only; no aggregate response metrics or outcome-driven selection.",
    "wall_seconds": time.perf_counter() - started,
}
(ROOT / "results/independent-admission.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, sort_keys=True))
