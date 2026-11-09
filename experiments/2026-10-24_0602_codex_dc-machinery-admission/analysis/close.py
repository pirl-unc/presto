"""Verify retrieval artifacts and record why quantitative admission is pending."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

experiment = Path(__file__).resolve().parents[1]
config = json.loads((experiment / "reproduce/launch.json").read_text())
root = Path(config["source_root"]) / "retrieval"
rows = []
for path in sorted(root.glob("*/receipt.json")):
    receipt = json.loads(path.read_text())
    body = path.parent / "body"
    if "sha256" in receipt:
        assert body.stat().st_size == receipt["n_bytes"]
        assert hashlib.sha256(body.read_bytes()).hexdigest() == receipt["sha256"]
    else:
        assert "error" in receipt
    copy = experiment / f"results/retrieval-{path.parent.name}.json"
    assert json.loads(copy.read_text()) == receipt
    rows.append({"attempt": path.parent.name, **receipt})
metadata = json.loads((root / "europepmc-metadata/body").read_text())
article = metadata["resultList"]["result"][0]
assert article["pmcid"] == config["source_pmcid"]
assert article["isOpenAccess"] == "N" and article["hasSuppl"] == "Y"
listing = ET.parse(root / "pmc-cloud-list/body").getroot()
assert listing.findtext("{*}KeyCount") == "0"
assert listing.findtext("{*}IsTruncated") == "false"
error = ET.parse(root / "europepmc-supplements/body").getroot()
message = error.findtext("errMsg")
assert "not open access" in message
summary = {
    "source_pmid": config["source_pmid"],
    "source_pmcid": config["source_pmcid"],
    "n_retrieval_attempts": len(rows),
    "n_hashed_bodies_verified": sum("sha256" in row for row in rows),
    "n_quantitative_assets_admitted": 0,
    "n_training_observations": 0,
    "europepmc_supplement_error": message,
    "europepmc_metadata": {key: article[key] for key in ["isOpenAccess", "hasSuppl"]},
    "pmc_cloud_n_keys": 0,
    "decision": "Quantitative admission pending an accessible original supplement; no fit",
    "next": "Existing Hitlist726 HAP1 quantitative source; retain DC source request798",
    "elapsed_http_seconds": sum(row["elapsed_seconds"] for row in rows),
    "pypi_2_0_17_status": "Official JSON endpoint returns404; deployment not verified",
    "verification": "Every retained body size/hash and receipt copy agrees",
}
(experiment / "results/access-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
