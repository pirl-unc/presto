"""Verify the post-audit standard summary and public-source receipts for closure."""

import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1]
D = Path(json.loads((P / "reproduce/launch.json").read_text())["source_directory"])
rows = json.loads((P / "results/standard-source-rows.json").read_text())
audit = json.loads((P / "results/source-audit.json").read_text())
summary = json.loads((P / "results/standard-summary.json").read_text())
assert len(rows) == summary["standard_query_rows"] == 90
assert sum(float(x["intensity"]) > 0 for x in rows) == summary["positive_intensities"] == 88
assert sum(float(x["intensity"]) == 0 for x in rows) == summary["literal_zero_intensities"] == 2
assert sum(x["m_score"] == "NA" for x in rows) == summary["missing_m_score"] == 3
assert [x["canonical_candidate"] for x in audit["nominal_standards"] if not x["observed_sequence_forms"]] == summary["absent_panel_standards"]
for standard in audit["nominal_standards"]:
    selected = [x for x in rows if x["canonical_candidate"] == standard["canonical_candidate"]]
    expected_sequence = standard["reported_form"].replace("[L(13C6; 15N)]", "L(UniMod:695)")
    expected_group = standard["reported_form"].replace("[L(13C6; 15N)]", "L(Label:13C(6)15N(1))") + "_2"
    assert all(x["sequence"] == expected_sequence and x["group_label"] == expected_group and x["charge"] == "2" for x in selected)
for record in json.loads((P / "results/input-manifest.json").read_text())["assets"]:
    content = (D / record["path"]).read_bytes()
    assert len(content) == record["bytes"]
    assert hashlib.sha256(content).hexdigest() == record["sha256"]
for name in ["pipeline-source-retrieval.json", "pipeline-dialignr-retrieval.json"]:
    record = json.loads((P / "reproduce" / name).read_text())
    content = Path(record["local_path"]).read_bytes()
    blob = hashlib.sha1(("blob " + str(len(content)) + "\0").encode() + content).hexdigest()
    assert blob == record["git_blob_sha_verified"] == record["result"]["sha"]
result = {"passed": True, "standard_rows": len(rows), "whole_source_assets_verified": len(json.loads((P / "results/input-manifest.json").read_text())["assets"]), "pipeline_blobs_verified": 2, "new_model_or_fit": False}
target = P / "results/closure-verification.json"
assert not target.exists(), "Do not overwrite closure receipts"
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
