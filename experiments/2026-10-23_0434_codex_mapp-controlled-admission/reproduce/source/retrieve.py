"""Pinned public metadata/article retrieval; no raw acquisition downloads."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, time, hashlib, urllib.request, urllib.error, traceback, datetime, shutil

R = Path(__file__).resolve().parents[1]
C = json.loads((R / "reproduce/launch.json").read_text())
A = Path(C["additional_assets"])
E = Path(C["existing_assets"])
receipts = []


def fetch(relative, url):
    p = A / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    r = {
        "path": relative,
        "url": url,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        if p.exists():
            raise FileExistsError(str(p))
        req = urllib.request.Request(
            url, headers={"User-Agent": "Presto-source-admission/1.0"}
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
            r.update(
                http_status=response.status,
                content_type=response.headers.get("Content-Type"),
                final_url=response.url,
            )
        p.write_bytes(data)
        r.update(
            passed=True, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
        )
    except Exception as error:
        r.update(passed=False, error=type(error).__name__ + ": " + str(error))
    r["wall_seconds"] = time.perf_counter() - start
    print(json.dumps(r), flush=True)
    return r


jobs = []
for project in C["source_projects"]:
    for leaf in ("project.json", "files.json"):
        relative = project + "/" + leaf
        if (E / relative).exists():
            data = (E / relative).read_bytes()
            dest = A / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            receipts.append(
                {
                    "path": relative,
                    "reused_existing_source": str(E / relative),
                    "passed": True,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        else:
            url = (
                "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/"
                + project
                + ("" if leaf == "project.json" else "/files?pageSize=100&page=0")
            )
            jobs.append((relative, url))
jobs += [
    (
        "article.xml",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8897557/fullTextXML",
    ),
    (
        "supplements.zip",
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8897557/supplementaryFiles",
    ),
]
with ThreadPoolExecutor(max_workers=3) as executor:
    for result in executor.map(lambda x: fetch(*x), jobs):
        receipts.append(result)
        (R / "results/retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
for project in C["source_projects"]:
    file = A / project / "files.json"
    if not file.exists():
        continue
    rows = json.loads(file.read_text())
    assert isinstance(rows, list)
    page = 1
    while len(rows) == 100:
        result = fetch(
            project + "/files-page" + str(page) + ".json",
            "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/"
            + project
            + "/files?pageSize=100&page="
            + str(page),
        )
        receipts.append(result)
        if not result["passed"]:
            break
        rows = json.loads((A / result["path"]).read_text())
        page += 1
        assert page < 20
(R / "results/retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
