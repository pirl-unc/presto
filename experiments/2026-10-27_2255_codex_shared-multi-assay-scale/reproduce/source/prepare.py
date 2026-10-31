"""Freeze the common runtime and six-output manifests before predictive fits."""

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
COHORT = ROOT / "data/modular_cohorts/2026-10-27-purified-multi-assay"
UPLOAD = ROOT / "artifacts" / FAMILY.name / "upload"
VARIANTS = ("ic50-exact", "ic50-bounds", "multi-assay", "multi-assay-wide")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    verification = (
        ROOT
        / "experiments/2026-10-27_2235_codex_shared-support-audit/results/admission-independent-verification.json"
    )
    assert json.loads(verification.read_text())["status"] == "passed"
    if (FAMILY / "reproduce/source-pins.json").exists():
        raise ValueError("already frozen")
    cohort = json.loads((COHORT / "manifest.json").read_text())
    old = json.loads(
        (
            ROOT
            / "experiments/2026-10-27_2100_codex_transfer-improvements/reproduce/source-pins.json"
        ).read_text()
    )
    files = sorted(
        set(old["runtimes"]["interaction"]["files"]) | {"data/modular_hitlist.py"}
    )
    runtime = UPLOAD / "runtimes/shared/presto"
    for rel in files:
        destination = runtime / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    sources = []
    census = Counter()
    train_lengths = Counter()
    pairs = set()
    peptides = set()
    for spec in cohort["sources"]:
        if Path(spec["path"]).name.startswith("koff-"):
            continue
        source = COHORT / spec["path"]
        assert sha(source) == spec["sha256"]
        destination = UPLOAD / "inputs/data" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources.append({**spec, "path": "data/" + source.name})
        for line in source.open():
            row = json.loads(line)
            m = row["mhc"][0]
            census[
                (row["endpoint"], row["split"], m["mhc_class"], row["qualifier"])
            ] += 1
            if row["split"] == "train":
                train_lengths[(m["mhc_class"], len(row["peptide"]))] += 1
                pairs.add((m["groove1"], m["groove2"]))
                peptides.add(row["peptide"])
    pins = dict(
        source_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        source_git_dirty=subprocess.check_output(
            ["git", "status", "--short"], text=True, cwd=ROOT
        ),
        runtimes={
            "shared": dict(
                local_runtime=str(runtime),
                files={rel: sha(runtime / rel) for rel in files},
            )
        },
        conditions={},
        manifests={},
        input_sources=sources,
        independent_verification_sha256=sha(verification),
        cohort_manifest_sha256=sha(COHORT / "manifest.json"),
        census=[
            dict(endpoint=k[0], split=k[1], mhc_class=k[2], qualifier=k[3], n=v)
            for k, v in sorted(census.items())
        ],
        training_input_diversity=dict(
            peptides=len(peptides),
            groove_pairs=len(pairs),
            lengths=[
                dict(mhc_class=k[0], length=k[1], n=v)
                for k, v in sorted(train_lengths.items())
            ],
        ),
        primary_original_ic50_files={
            s: f"data/ic50-original-{s}.jsonl" for s in ("train", "validation", "test")
        },
    )
    for seed in (42, 43):
        for variant in VARIANTS:
            name = f"{variant}-s{seed}"
            chosen = [
                s
                for s in sources
                if variant.startswith("multi-")
                or Path(s["path"]).name.startswith("ic50-")
                and (variant == "ic50-bounds" or "original-" in s["path"])
            ]
            model = {
                **cohort["model"],
                "endpoints": [
                    "ec50",
                    "half_life",
                    "ic50",
                    "kd",
                    "kd_via_ec50",
                    "kd_via_ic50",
                ]
                if variant.startswith("multi-")
                else ["ic50"],
            }
            if variant.endswith("-wide"):
                model.update(d_model=192, n_layers=3, n_heads=6, d_ff=768)
            training = dict(
                seed=seed,
                batch_size=64,
                max_steps=6000,
                max_epochs=10000,
                min_epochs=10000,
                patience=10000,
                identity_weight=0,
                learning_rate=3e-4,
                weight_decay=1e-4,
                checkpoint_every_steps=100,
                validation_every_steps=500,
                balance_by=[],
            )
            manifest = dict(
                schema=cohort["schema"], model=model, training=training, sources=chosen
            )
            pins["conditions"][name] = dict(
                runtime="shared",
                classes=["I", "II"],
                seed=seed,
                max_steps=6000,
                variant=variant,
                balance_by=[],
            )
            pins["manifests"][name] = manifest
            write(UPLOAD / "inputs" / f"{name}.json", manifest)
            write(FAMILY / "reproduce/manifests" / f"{name}.json", manifest)
    for dest in (UPLOAD / "source-pins.json", FAMILY / "reproduce/source-pins.json"):
        write(dest, pins)
    shutil.copytree(UPLOAD / "runtimes", FAMILY / "reproduce/source/runtimes")
    shutil.copy2(__file__, FAMILY / "reproduce/source/prepare.py")
    print(
        json.dumps(
            dict(
                conditions=list(pins["conditions"]),
                upload_bytes=sum(
                    p.stat().st_size for p in UPLOAD.rglob("*") if p.is_file()
                ),
                census=pins["census"],
                diversity=pins["training_input_diversity"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
