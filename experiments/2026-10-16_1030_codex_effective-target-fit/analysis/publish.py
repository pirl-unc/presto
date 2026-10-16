"""Prepare, publish and independently verify the registered qualification evidence."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from datacache import install_bundle, materialize

FAMILY = Path(__file__).resolve().parents[1]
REPOSITORY = FAMILY.parents[1]
TAG = "effective-target-fit-qualification-v1"
RELEASE = "https://github.com/pirl-unc/presto/releases/tag/" + TAG
BASE = "https://github.com/pirl-unc/presto/releases/download/" + TAG + "/"
CACHE = Path.home() / ".hitlist/presto-effective-target-fit"


def pin(path):
    path = Path(path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def prepare():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
    ), "Commit the reviewed analysis and narrative before publication preparation"
    run = json.loads((FAMILY / "results/fit.json").read_text())
    analysis = json.loads((FAMILY / "results/analysis-receipt.json").read_text())
    assert run["summary"]["status"] == "success"
    files = {name: Path(path) for name, path in run["local_paths"].items()}
    for name, path in files.items():
        assert pin(path) == {
            "sha256": run["files"][name]["sha256"],
            "size": run["files"][name]["size_bytes"],
        }
    for name, record in analysis["analysis_outputs"].items():
        files[name] = Path(record["path"])
        assert pin(files[name]) == {
            "sha256": record["sha256"],
            "size": record["size_bytes"],
        }
    species = json.loads((FAMILY / "results/species-audit-receipt.json").read_text())
    for name, record in species.items():
        files[name] = Path(record["path"])
        assert pin(files[name]) == {
            "sha256": record["sha256"],
            "size": record["size_bytes"],
        }
    for name, relative in {
        "README.md": "README.md",
        "protocol-notes.json": "results/protocol-notes.json",
        "verify.py": "analysis/verify.py",
    }.items():
        files[name] = FAMILY / relative
    contract = json.loads((FAMILY / "results/contract-publication.json").read_text())
    files["original-plan.md"] = Path(contract["downloaded"]["plan.md"])
    inventory = json.loads(files["inventory.json"].read_text())
    exposed = [
        "membership.tsv",
        "selected-source-contexts.tsv",
        "supported_outputs.json",
        "output_coverage.csv",
        "split_support.csv",
        "affinity_target_audit.json",
        "affinity_target_changes.tsv",
        "model-construction-policy-check.json",
        "uncertainty.json",
        "decision.json",
        "baseline-state.json",
        "output_updates.json",
        "regularization.json",
    ]
    assert all(name in inventory for name in exposed)
    reproduce = {
        p.relative_to(FAMILY).as_posix(): p
        for p in sorted((FAMILY / "reproduce").rglob("*"))
        if p.is_file()
    }
    reproduce.update(
        {
            "analysis/verify.py": FAMILY / "analysis/verify.py",
            "analysis/publish.py": Path(__file__),
            "analysis/audit_species.py": FAMILY / "analysis/audit_species.py",
            "results/fit.json": FAMILY / "results/fit.json",
            "results/protocol-notes.json": FAMILY / "results/protocol-notes.json",
            "manifest.json": FAMILY / "manifest.json",
            "README.md": FAMILY / "README.md",
        }
    )
    sources = {name: {"path": str(path), **pin(path)} for name, path in files.items()}
    sources.update(
        {
            "reproduce/" + name: {"path": str(path), **pin(path)}
            for name, path in reproduce.items()
        }
    )
    names = [*files, *exposed, "reproduce.tar.gz", "publication.json", "SHA256SUMS"]
    assert len(set(names)) == len(names)

    def build(inputs, outputs):
        for name in files:
            shutil.copyfile(inputs[name], outputs[name])
        with tarfile.open(inputs["run.tar.gz"]) as archive:
            members = archive.getmembers()
            assert len(members) == len(inventory)
            assert {m.name for m in members} == set(inventory)
            for member in members:
                assert member.isfile()
                data = archive.extractfile(member).read()
                assert inventory[member.name] == {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                if member.name in exposed:
                    Path(outputs[member.name]).write_bytes(data)
        with tarfile.open(outputs["reproduce.tar.gz"], "w:gz") as archive:
            for name in reproduce:
                archive.add(inputs["reproduce/" + name], arcname=name, recursive=False)
        assets = {
            name: {"url": BASE + name, **pin(outputs[name])}
            for name in names
            if name not in {"publication.json", "SHA256SUMS"}
        }
        write(
            outputs["publication.json"],
            {
                "schema_version": 1,
                "release": RELEASE,
                "executed_recipe_commit": run["summary"]["commit"],
                "analysis_publication_commit": commit,
                "producer": run["summary"]["installed_producer"],
                "publication_script_sha256": pin(__file__)["sha256"],
                "n_inventory_verified_archive_members": len(inventory),
                "decision": json.loads(Path(outputs["decision.json"]).read_text()),
                "files": assets,
                "archive_contents": "Complete predictions, checkpoints, logs, loss ledgers, source context, membership, leakage and target audits; see inventory.json.",
                "inspection": "ic50-inspection.tsv joins all 2,614 measured IC50 rows to selected predictions, original/effective targets, source citations and cell/MHC/condition/assay context. Full native context and original frozen membership are separate direct downloads.",
                "limitations": "Exploratory peptide-disjoint evaluation; shared studies and unknown donor/specimen/acquisition identities. Eight elution rows record rat hosts with human HLA; the consumer conflates host and MHC species (#115). See species-audit.json and source-species.tsv. All other outputs remain undeclared.",
            },
        )
        Path(outputs["SHA256SUMS"]).write_text(
            "".join(
                pin(outputs[name])["sha256"] + "  " + name + "\n"
                for name in names
                if name != "SHA256SUMS"
            )
        )

    paths = materialize(
        CACHE / "publication" / commit,
        sources,
        transform={
            "version": "effective-target-fit-publication-v1",
            "options": {"commit": commit, "script_sha256": pin(__file__)["sha256"]},
        },
        outputs={name: {} for name in names},
        builder=build,
    )
    receipt = {
        "tag": TAG,
        "release": RELEASE,
        "commit": commit,
        "files": {
            name: {"url": BASE + name, **pin(path)} for name, path in paths.items()
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }
    write(FAMILY / "results/publication.json", receipt)
    print("Prepared", len(paths), "verified assets; publication remains pending.")


def publish():
    receipt = json.loads((FAMILY / "results/publication.json").read_text())
    for name, path in receipt["paths"].items():
        assert pin(path) == {
            key: receipt["files"][name][key] for key in ("sha256", "size")
        }
    notes = FAMILY / "results/release-notes.md"
    notes.write_text(
        "Frozen IC50 qualification with explicit effective targets and separate raw-measurement diagnostics. "
        "Download README.md for the observed outcome and limitations, ic50-inspection.tsv for source-linked measurements/predictions, "
        "and run.tar.gz with inventory.json for the complete preserved run. "
        "Includes exact membership, target policies, output coverage, uncertainty, checkpoints, loss ledgers, reproducibility sources and SHA-256 hashes. "
        "Unknown biological identity remains a limitation.\n"
    )
    gh(
        "release",
        "create",
        TAG,
        "--repo",
        "pirl-unc/presto",
        "--target",
        receipt["commit"],
        "--title",
        "Frozen IC50 fit: effective and raw target qualification v1",
        "--draft",
        "--notes-file",
        str(notes),
        *receipt["paths"].values(),
    )
    release = next(
        r
        for r in json.loads(gh("api", "repos/pirl-unc/presto/releases?per_page=100"))
        if r["tag_name"] == TAG
    )
    assert release["draft"] and {a["name"] for a in release["assets"]} == set(
        receipt["files"]
    )
    for asset in release["assets"]:
        expected = receipt["files"][asset["name"]]
        assert asset["size"] == expected["size"]
        assert asset["digest"] == "sha256:" + expected["sha256"]
    gh(
        "api",
        "--method",
        "PATCH",
        f"repos/pirl-unc/presto/releases/{release['id']}",
        "-F",
        "draft=false",
        "-f",
        "make_latest=false",
    )
    verify_public(receipt)


def verify_public(receipt):
    downloaded = install_bundle(
        CACHE / "public-verification" / TAG,
        receipt["files"],
        download_options={"show_progress": False},
    )
    for name, path in downloaded.items():
        assert Path(path).read_bytes() == Path(receipt["paths"][name]).read_bytes()
    write(
        FAMILY / "results/public-verification.json",
        {
            "release": RELEASE,
            "n_verified_public_assets": len(downloaded),
            "files": receipt["files"],
            "downloaded": {n: str(p) for n, p in downloaded.items()},
        },
    )
    print("Independently downloaded and verified", len(downloaded), "assets:", RELEASE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "publish", "verify"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "publish":
        publish()
    else:
        verify_public(json.loads((FAMILY / "results/publication.json").read_text()))
