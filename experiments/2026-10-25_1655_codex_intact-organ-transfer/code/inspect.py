"""Metadata-only F3 feasibility through the canonical public evidence interface."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import hitlist
from hitlist.evidence_store import EvidenceStore


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    checkout = Path(config["hitlist_checkout"])
    assert Path(hitlist.__file__).resolve().is_relative_to(checkout)
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == config["hitlist_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=checkout,
        text=True,
    ).strip()
    package = Path(config["source_package"])
    assert (
        hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
        == config["source_manifest_sha256"]
    )
    store = EvidenceStore.open(package)
    queried = {}
    for name, columns in request["queries"].items():
        assert not set(columns) & {
            "quantitative_value",
            "quantitative_raw",
            "details_json",
        }
        queried[name] = store.query(name, select=columns, limit=None).table.to_pylist()
    views = {v["view_id"]: v for v in queried["intact_source_views"]}
    acquisitions = {a["acquisition_id"]: a for a in queried["intact_acquisitions"]}
    roles = defaultdict(set)
    for view in views.values():
        if view["acquisition_id"]:
            roles[view["source_view_role"]].add(view["acquisition_id"])
    old_ids = roles["kinetics"]
    old_raw = {acquisitions[i]["raw_file"] for i in old_ids if i in acquisitions}
    reused_views = set()
    reuse_claims = []
    for claim in queried["intact_reconciliations"]:
        if claim["assertion_predicate"] != "repeated_acquisition_source_views":
            continue
        left, right = (
            views.get(claim[k], {}) for k in ("source_subject_id", "target_subject_id")
        )
        if {left.get("source_view_role"), right.get("source_view_role")} == {
            "kinetics",
            "organs",
        }:
            reuse_claims.append(claim)
            reused_views.update(
                v["view_id"] for v in (left, right) if v["source_view_role"] == "organs"
            )
    decisions, selected_ids = [], set()
    for view in views.values():
        if view["source_view_role"] != "organs":
            continue
        acquisition = acquisitions.get(view["acquisition_id"], {})
        reasons = []
        if not acquisition or not acquisition.get("raw_file"):
            reasons.append("unresolved_acquisition")
        if view["acquisition_id"] in old_ids or acquisition.get("raw_file") in old_raw:
            reasons.append("F2_acquisition_overlap")
        if view["view_id"] in reused_views:
            reasons.append("source_backed_reused_view")
        if acquisition.get("organ_label") not in config["required_organs"]:
            reasons.append("outside_fixed_organ_scope")
        if not reasons:
            selected_ids.add(view["acquisition_id"])
        decisions.append({"view": view, "acquisition": acquisition, "reasons": reasons})
    counts = Counter(
        (acquisitions[i]["organ_label"], acquisitions[i]["time_minutes"])
        for i in selected_ids
    )
    support = [
        {"organ": organ, "minutes": time, "n_acquisitions": counts[organ, time]}
        for organ in config["required_organs"]
        for time in [0.0, 30.0, 60.0, 90.0, 120.0]
    ]
    gate = all(
        counts[organ, time] > 0
        for organ in config["required_organs"]
        for time in [0.0, 30.0, 60.0]
    )
    summary = {
        "status": "metadata_supported_numeric_admission_pending"
        if gate
        else "metadata_gate_failed",
        "metadata_gate_passed": gate,
        "source_manifest_sha256": config["source_manifest_sha256"],
        "hitlist_commit": config["hitlist_commit"],
        "n_queried_rows": {k: len(v) for k, v in queried.items()},
        "n_F2_acquisitions": len(old_ids),
        "n_F3_views": len(decisions),
        "n_selected_F3_acquisitions": len(selected_ids),
        "n_F2_F3_reuse_claims": len(reuse_claims),
        "n_shared_acquisition_ids": len(old_ids & roles["organs"]),
        "excluded_reasons": dict(
            Counter(reason for d in decisions for reason in d["reasons"])
        ),
        "coverage": support,
        "numeric_outcomes_queried": False,
        "physical_independence": "unresolved; one source animal and one substrate",
        "next_action": "Qualify shared F3 zero-baseline endpoint before any predictions"
        if gate
        else "Stop; do not substitute organs or times",
    }
    write(root / "results/canonical-metadata.json", queried)
    write(root / "results/acquisition-decisions.json", decisions)
    write(root / "results/metadata-support.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
