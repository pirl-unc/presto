"""Offline source-byte replay and independent supplement metadata/table checks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

from bs4 import BeautifulSoup
from pypdf import PdfReader
import psutil

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
RAW = ROOT / "data/endolysosomal_source_audit" / FAMILY.name


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    ledger = read(FAMILY / "results/discovery-ledger.json")
    config = read(FAMILY / "reproduce/launch.json")
    amendment = read(FAMILY / "reproduce/retrieval-amendment.json")
    assert config["network_operation_limit"] == amendment["original_limit"] == 12
    assert len(ledger["operations"]) == ledger["limit"] == amendment["amended_limit"] == 15
    assert [r["number"] for r in ledger["operations"]] == list(range(1, 16))
    assert amendment["operations_already_used"] == 9
    assert datetime.fromisoformat(amendment["utc"]) < datetime.fromisoformat(
        ledger["operations"][9]["utc"]
    )
    pins = {}
    for item in ledger["operations"]:
        if item.get("tool") == "urllib":
            assert item["status"] == "completed" and item["status_code"] == 200
            path = ROOT / item["path"]
            assert sha(path) == item["sha256"]
            assert path.stat().st_size == item["bytes"] <= 32 * 1024**2
            assert item["elapsed_seconds"] < 60
            pins[item["path"]] = item["sha256"]
        elif item.get("response_path"):
            path = ROOT / item["response_path"]
            assert sha(path) == item["response_sha256"]
            pins[item["response_path"]] = item["response_sha256"]
    local_bytes = sum(r.get("bytes", 0) for r in ledger["operations"])
    assert local_bytes < 64 * 1024**2
    last = ledger["operations"][-1]
    retrieval_span = (
        datetime.fromisoformat(last["utc"]) - datetime.fromisoformat(config["utc"])
    ).total_seconds() + last["elapsed_seconds"]
    assert retrieval_span < 600

    html = BeautifulSoup((RAW / "article.html").read_text(), "html.parser")
    nodes = json.loads(html.find("script", id="__NUXT_DATA__").string)
    file_objects = [n for n in nodes if isinstance(n, dict) and "fileServerId" in n]
    published = sorted(
        nodes[n["name"]]
        for n in file_objects
        if nodes[n["fileServerId"]].startswith("1936335/data-sheet/")
    )
    api = read(RAW / "supplements.json")
    assert published == sorted(r["name"] for r in api)
    assert len(api) == 5
    inventory = []
    texts = {}
    for n in range(1, 6):
        path = RAW / f"Data_Sheet_{n}.pdf"
        reader = PdfReader(path)
        assert len(reader.pages) == 1 and not list(reader.attachments)
        info = subprocess.check_output(["pdfinfo", str(path)], text=True, timeout=60)
        assert re.search(r"Pages:\s+1\b", info)
        text = subprocess.check_output(
            ["pdftotext", "-layout", str(path), "-"], text=True, timeout=60
        )
        texts[n] = text
        inventory.append(
            {
                "file": path.name,
                "sha256": sha(path),
                "pages": 1,
                "embedded_files": 0,
                "poppler_text_chars": len(text),
                "poppler_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    # Preserve the original aggregate table; do not expand means into individual labels.
    expected = {
        "Number of values": [7.0, 4.0],
        "Mean": [1.0, 16.0],
        "Std. Deviation": [0.28, 3.3],
        "Std. Error of Mean": [0.1, 1.6],
    }
    pypdf_table = PdfReader(RAW / "Data_Sheet_2.pdf").pages[0].extract_text()
    for label, values in expected.items():
        for text in (texts[2], pypdf_table):
            matched = re.search(re.escape(label) + r"\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", text)
            assert matched and list(map(float, matched.groups())) == values
    timepoints = {}
    for n, expected_times in (
        (3, [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 16.0, 24.0]),
        (4, [0.0, 3.0, 8.0, 16.0, 24.0, 48.0, 60.0, 72.0, 96.0]),
    ):
        times = sorted(set(map(float, re.findall(r"(?<!\S)(\d+(?:\.\d+)?)h(?!\S)", texts[n]))))
        assert times == expected_times, (n, times)
        assert "Intensity (Log2)" in texts[n]
        timepoints[str(n)] = times
    assert psutil.virtual_memory().available >= config["available_ram_floor"]
    assert shutil.disk_usage(ROOT).free >= config["disk_floor"]
    own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    assert max(own_peak, child_peak) < config["local_rss_limit_bytes"]
    assert time.monotonic() - start < 900
    result = dict(
        status="passed",
        utc=datetime.now(timezone.utc).isoformat(),
        source_pins=pins,
        original_asset_count=6,
        supplementary_pages=5,
        inventory=inventory,
        aggregate_half_life_table_as_reported=expected,
        table_unit="hours except counts",
        timepoint_labels_hours=timepoints,
        peptide_quantity_rows_admitted=0,
        full_model_qualified=False,
        retrieval_seconds=retrieval_span,
        local_http_bytes=local_bytes,
        verification_seconds=time.monotonic() - start,
        process_peak_rss_bytes=own_peak,
        maximum_child_rss_bytes=child_peak,
        resource_note=(
            "Post-execution peak RSS checks; Poppler timeout60s per call. "
            "Initial ad hoc pypdf extraction had unmeasured runtime/RSS and capped "
            "form extraction; not used to assert text completeness."
        ),
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("source_pins", "inventory")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
