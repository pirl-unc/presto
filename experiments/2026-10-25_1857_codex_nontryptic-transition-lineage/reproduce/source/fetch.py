"""Bounded anonymous public retrieval with immutable request and response receipts."""

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.links = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.current = {"url": urljoin(self.base, attrs["href"]), "label": ""}
            self.links.append(self.current)

    def handle_endtag(self, tag):
        if tag == "a":
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current["label"] += data


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "reproduce/launch.json").read_text())
    request_set = json.loads(args.requests.read_text())
    phase = request_set["phase"]
    assert phase in ("metadata", "processed")
    destination = Path(cfg["data_directory"])
    destination.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(p.read_text()) for p in destination.glob("*.receipt.json")]
    if phase == "metadata":
        assert (
            sum(r["phase"] == "metadata" for r in prior) + len(request_set["requests"])
            <= cfg["network_limits"]["metadata_requests"]
        )
    total = sum(p.stat().st_size for p in destination.iterdir() if p.is_file())
    all_passed = True
    for request in request_set["requests"]:
        name = request["name"]
        assert Path(name).name == name and not name.startswith(".")
        url = request["url"]
        assert (
            urlparse(url).scheme == "https"
            and urlparse(url).hostname in cfg["public_hosts"]
        )
        assert request["discovery_provenance"] and request["expected_kind"] in (
            "html",
            "xml",
            "json",
            "xlsx",
            "pdf",
            "zip",
            "text",
        )
        receipt_path = destination / (name + ".receipt.json")
        response_path = destination / name
        partial = destination / (name + ".part")
        if any(p.exists() for p in (receipt_path, response_path, partial)):
            raise FileExistsError("Preserve previous request: " + name)
        started = time.monotonic()
        receipt = {
            "request": request,
            "phase": phase,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "request_set": str(args.requests.resolve()),
            "bytes": 0,
            "status": "started",
        }
        digest = hashlib.sha256()
        limit = cfg["network_limits"][
            "metadata_bytes_each" if phase == "metadata" else "processed_bytes_each"
        ]
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(
                    cfg["network_limits"]["connect_seconds"],
                    cfg["network_limits"]["read_seconds"],
                ),
                headers={"User-Agent": "Presto-public-source-audit/1.0"},
            ) as response:
                receipt.update(
                    http_status=response.status_code,
                    final_url=response.url,
                    content_type=response.headers.get("Content-Type"),
                    content_length=response.headers.get("Content-Length"),
                    redirects=[
                        {"status": r.status_code, "url": r.url}
                        for r in response.history
                    ],
                )
                length = response.headers.get("Content-Length")
                if length and int(length) > limit:
                    raise ValueError("declared_asset_size_limit")
                with partial.open("xb") as stream:
                    for chunk in response.iter_content(128 * 1024):
                        if (
                            time.monotonic() - started
                            > cfg["network_limits"]["request_wall_seconds"]
                        ):
                            raise TimeoutError("request_wall_time")
                        if (
                            receipt["bytes"] + len(chunk) > limit
                            or total + len(chunk) > cfg["network_limits"]["total_bytes"]
                        ):
                            raise ValueError("retrieval_byte_budget")
                        if (
                            shutil.disk_usage(destination).free
                            < cfg["limits"]["disk_floor_bytes"]
                        ):
                            raise ValueError("disk_reserve")
                        stream.write(chunk)
                        digest.update(chunk)
                        receipt["bytes"] += len(chunk)
                        total += len(chunk)
                partial.rename(response_path)
                response.raise_for_status()
            receipt["sha256"] = digest.hexdigest()
            kind = request["expected_kind"]
            if kind in ("html", "xml", "json", "text"):
                encoding = request.get("encoding", "utf-8")
                assert encoding in ("utf-8", "latin-1")
                raw = response_path.read_bytes()
                text = raw.decode(encoding)
                assert text.encode(encoding) == raw
                receipt["text_decoding"] = encoding
                if "checking your browser" in text[:5000].lower():
                    raise ValueError("html_browser_challenge_not_dataset")
                if kind == "xml":
                    root = ET.fromstring(text)
                    receipt["xml_root"] = root.tag
                    links = [
                        {
                            "url": urljoin(receipt["final_url"], v),
                            "label": "".join(e.itertext())[:200],
                        }
                        for e in root.iter()
                        for k, v in e.attrib.items()
                        if k.endswith("href")
                    ]
                elif kind == "html":
                    if "<html" not in text.lower():
                        raise ValueError("not_html")
                    parser = Links(receipt["final_url"])
                    parser.feed(text)
                    links = parser.links
                elif kind == "json":
                    value = json.loads(text)
                    receipt["json_top_keys"] = (
                        list(value) if isinstance(value, dict) else None
                    )
                    links = []
                else:
                    links = []
                if phase == "metadata":
                    write(ROOT / "results" / (name + ".links.json"), links)
            elif kind == "pdf":
                if not response_path.read_bytes().startswith(b"%PDF-"):
                    raise ValueError("not_pdf")
            elif not response_path.read_bytes()[:4].startswith(b"PK"):
                raise ValueError("not_zip_container")
            receipt["status"] = "retrieved_validated_payload"
        except Exception as error:
            receipt.update(
                status="failed",
                error=type(error).__name__
                + ": "
                + (
                    error.reason
                    if isinstance(error, UnicodeDecodeError)
                    else str(error)
                ),
                sha256=digest.hexdigest(),
            )
            all_passed = False
        receipt["elapsed_seconds"] = time.monotonic() - started
        write(receipt_path, receipt)
        write(ROOT / "results" / (name + ".receipt.json"), receipt)
        print(
            json.dumps({k: v for k, v in receipt.items() if k != "request"}), flush=True
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
