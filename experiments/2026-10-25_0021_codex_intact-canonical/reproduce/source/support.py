"""Descriptive kinetic support, no fitting or held-out performance selection."""

import collections
import json
import math
from pathlib import Path

from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
assert json.loads((ROOT / "results/independent-replay.json").read_text())["passed"]
store = EvidenceStore.open(config["package"])


def query(name):
    return store.query("intact_" + name, limit=None).table.to_pylist()


views = {r["view_id"]: r for r in query("source_views")}
acquisitions = {r["acquisition_id"]: r for r in query("acquisitions")}
quantities = {r["measurement_id"]: r for r in query("quantities")}
decisions = []
for row in query("assignments"):
    if views[row["view_id"]]["source_view_role"] != "kinetics":
        continue
    q = quantities[row["measurement_id"]]
    a = acquisitions[row["acquisition_id"]]
    reasons = []
    value = q["quantitative_value"]
    if value is None or not math.isfinite(value) or value <= 0:
        reasons.append("not_positive_finite")
    if row["span_status"] != "matches_source_substrate":
        reasons.append("span_conflict")
    if row["mass_link_status"] != "input_id_and_intensity_agree":
        reasons.append("input_link_or_quantity_conflict")
    if row["n_assignments_reported"] != 1 or row["n_assignments_replayed"] != 1:
        reasons.append("ambiguous_mass_assignment")
    if row["mass_error_ppm"] is None or abs(row["mass_error_ppm"]) > 10:
        reasons.append("outside_reported_mass_tolerance")
    if json.loads(row["modifications_json"]) != {"Localized Mod": "0", "Unlocalized Mod": "None", "Term Mod": "0"}:
        reasons.append("not_explicit_unmodified")
    decisions.append({"assignment_id": row["assignment_id"], "measurement_id": q["measurement_id"],
                      "acquisition_id": row["acquisition_id"], "peptide_sequence": row["peptide_sequence"],
                      "start_1based": row["start_1based"], "end_1based": row["end_1based"],
                      "time_minutes": a["time_minutes"], "replicate_label_raw": a["replicate_label_raw"],
                      "raw_intensity": value, "reasons": reasons})
groups = collections.defaultdict(list)
for row in decisions:
    if not row["reasons"]:
        groups[row["peptide_sequence"], row["acquisition_id"]].append(row)
for rows in groups.values():
    if len(rows) > 1:
        for row in rows:
            row["reasons"].append("multiple_masses_for_sequence_acquisition")
retained = [r for r in decisions if not r["reasons"]]
by_acquisition, by_sequence = collections.defaultdict(list), collections.defaultdict(list)
for row in retained:
    by_acquisition[row["acquisition_id"]].append(row)
    by_sequence[row["peptide_sequence"]].append(row)
acquisition_support = []
for key, rows in sorted(by_acquisition.items(), key=lambda x: (acquisitions[x[0]]["time_minutes"], acquisitions[x[0]]["replicate_label_raw"])):
    a = acquisitions[key]
    intact = [r for r in rows if r["start_1based"] == 1 and r["end_1based"] == 140]
    acquisition_support.append({"acquisition_id": key, "time_minutes": a["time_minutes"],
                                "replicate_label_raw": a["replicate_label_raw"], "n_candidate_peptides": len(rows),
                                "n_intact_substrate_rows": len(intact)})
sequence_support = []
for sequence, rows in sorted(by_sequence.items()):
    times = sorted({r["time_minutes"] for r in rows})
    early = [r for r in rows if r["time_minutes"] <= 80]
    late = [r for r in rows if r["time_minutes"] >= 100]
    sequence_support.append({"peptide_sequence": sequence, "length": len(sequence),
                             "n_observations": len(rows), "times_minutes": times,
                             "n_early_times": len({r["time_minutes"] for r in early}),
                             "n_late_times": len({r["time_minutes"] for r in late}),
                             "n_early_observations": len(early), "n_late_observations": len(late)})
summary = {"status": "descriptive_support_not_training_admission", "n_source_assignment_rows": len(decisions),
           "n_candidate_rows": len(retained), "n_candidate_sequences": len(by_sequence),
           "exclusion_reasons_nonexclusive": dict(collections.Counter(reason for r in decisions for reason in r["reasons"])),
           "n_sequences_by_n_times": dict(collections.Counter(len(r["times_minutes"]) for r in sequence_support)),
           "n_sequences_with_at_least_two_early_times_and_a_late_time": sum(r["n_early_times"] >= 2 and r["n_late_times"] >= 1 for r in sequence_support),
           "acquisition_support": acquisition_support, "sequence_support": sequence_support,
           "models_trained": None, "predictive_metrics": None}
(ROOT / "results/kinetic-support.json").write_text(json.dumps(summary, indent=2) + "\n")
(ROOT / "results/kinetic-candidate-decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
print(json.dumps({k: v for k, v in summary.items() if k not in {"sequence_support", "acquisition_support"}}), flush=True)
