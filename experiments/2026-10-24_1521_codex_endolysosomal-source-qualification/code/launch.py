"""Public primary-asset inventory only; no biological source curation or fitting."""

import argparse
import datetime
import hashlib
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/endolysosomal_source_qualification/2026-10-24_1521_oztemiz-v1"
ARTICLE = "https://www.frontiersin.org/journals/allergy/articles/10.3389/falgy.2026.1936335/full"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch(url, name):
    target = RAW / name
    assert not target.exists(), "Preserve previous retrieval; choose a new receipt name"
    started = time.perf_counter()
    receipt = {"url": url, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Presto-public-source-qualification/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(100_000_001)
            assert len(payload) <= 100_000_000, "Asset exceeds this inventory's retrieval cap"
            receipt.update(
                status=response.status, final_url=response.url, headers=dict(response.headers)
            )
        target.write_bytes(payload)
        receipt.update(
            path=str(target), size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    except (urllib.error.URLError, TimeoutError) as error:
        receipt.update(error=str(error), access_status="failed_not_absent")
    receipt["seconds"] = time.perf_counter() - started
    write(RAW / (name + ".receipt.json"), receipt)
    print(json.dumps(receipt), flush=True)
    return target if target.exists() else None


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append({"tag": tag, "attribute": key, "value": value})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-url")
    parser.add_argument("--asset-name")
    args = parser.parse_args()
    if args.asset_url:
        assert args.asset_name and Path(args.asset_name).name == args.asset_name
        fetch(args.asset_url, args.asset_name)
        return
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=False)
    for directory in ["reproduce", "reproduce/source", "results", "launch_logs"]:
        (ROOT / directory).mkdir(exist_ok=True)
    shutil.copyfile(__file__, ROOT / "reproduce/source/launch.py")
    shutil.copyfile(
        REPO / "experiments/agents/codex/plans/endolysosomal-source-qualification.md",
        ROOT / "reproduce/plan.md",
    )
    receipt = {
        "agent": "Codex / GPT-6",
        "python": sys.executable,
        "platform": platform.platform(),
        "argv": sys.argv,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "raw_directory": str(RAW),
        "article_url": ARTICLE,
        "training": "none",
        "evaluation": "source qualification; no validation/test predictions apply",
        "environment_overrides": {},
        "quantitative_admission": False,
    }
    write(ROOT / "reproduce/launch.json", receipt)
    write(ROOT / "manifest.json", receipt)
    (ROOT / "reproduce/launch.sh").write_text(
        f"#!/bin/sh\nset -eu\ncd '{REPO}'\n'{sys.executable}' '{ROOT}/code/launch.py'\n"
    )
    path = fetch(ARTICLE, "article.html")
    if path:
        text = path.read_text()
        links = Links()
        links.feed(text)
        # Inventory public URL strings embedded in publisher HTML/JSON; no API guessing.
        embedded = sorted(set(re.findall(r'https?[^\s<>"\\]+', text)))
        candidates = [
            u
            for u in embedded
            if any(
                s in u.lower()
                for s in [
                    "supplement",
                    "data_sheet",
                    ".pdf",
                    ".xlsx",
                    ".zip",
                    "pride",
                    "proteomecentral",
                ]
            )
        ]
        write(
            ROOT / "results/article-links.json",
            {"html_links": links.links, "embedded_asset_candidates": candidates},
        )
        print(json.dumps({"asset_candidates": candidates}), flush=True)
    write(
        ROOT / "results/retrieval-summary.json",
        {
            "seconds": time.perf_counter() - started,
            "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
            "article_retrieved": bool(path),
            "no_quantitative_admission": True,
        },
    )


if __name__ == "__main__":
    main()
