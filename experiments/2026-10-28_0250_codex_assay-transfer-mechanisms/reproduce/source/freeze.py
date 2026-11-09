"""Freeze complete common source, homogeneous source files and twelve manifests."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
UPLOAD = ROOT / "artifacts" / F.name / "upload"
VARIANTS = (
    "small-ic50",
    "wide-ic50",
    "wide-six-natural",
    "wide-six-class-balanced",
    "wide-six-primary-matched",
    "wide-six-primary-shuffled",
)
ENDPOINTS = ("ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50", "half_life")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensors_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith("heads.") and not name.startswith("heads.ic50."):
            continue
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    assert not (F / "reproduce/source-pins.json").exists(), "already frozen"
    dp = json.loads((F / "reproduce/data-pins.json").read_text())
    assert (
        json.loads((F / "results/data-independent-verification.json").read_text())[
            "status"
        ]
        == "passed"
    )
    input_dir = UPLOAD / "inputs/data"
    input_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    transport_ids = {}
    row_metadata = {}
    original_line_hashes, transport_line_hashes = Counter(), Counter()
    for spec in dp["files"]:
        original = Path(dp["data_directory"]) / spec["path"]
        assert sha(original) == spec["sha256"]
        streams, counts, ids_by_source = {}, Counter(), {}
        with original.open("rb") as reader:
            for line in reader:
                row = json.loads(line)
                source = row["source"]
                assert source in ("hitlist-v4-iedb", "hitlist-v4-cedar")
                if source not in streams:
                    name = original.stem + "-" + source + ".jsonl"
                    streams[source] = (input_dir / name).open("xb")
                    ids_by_source[source] = []
                streams[source].write(line)
                counts[source] += 1
                ids_by_source[source].append(row["observation_id"])
                original_line_hashes[hashlib.sha256(line).hexdigest()] += 1
                if row["split"] == "train" and spec["variant"] == "measured":
                    row_metadata[row["observation_id"]] = SimpleNamespace(
                        endpoint=row["endpoint"],
                        mhc=(
                            SimpleNamespace(
                                mhc_class=row["mhc"][0]["mhc_class"],
                                class_override=None,
                            ),
                        ),
                    )
        for source, stream in streams.items():
            stream.close()
            path = Path(stream.name)
            transport_ids["data/" + path.name] = ids_by_source[source]
            for line in path.open("rb"):
                transport_line_hashes[hashlib.sha256(line).hexdigest()] += 1
            sources.append(
                dict(
                    path="data/" + path.name,
                    sha256=sha(path),
                    source=source,
                    population="new component split of earlier purified human six-output TRAIN rows",
                    label_policy=(
                        "synthetic TRAIN-only within-endpoint/class/groove/qualifier label permutation; never biological evidence"
                        if spec["variant"] == "shuffled"
                        else "original physical quantitative measurements and bounds; no clipping; training-only endpoint scales"
                    ),
                    split_policy=dp["split_salt"],
                    endpoint=spec["endpoint"],
                    split=spec["split"],
                    variant=spec["variant"],
                    rows=counts[source],
                    parent_prepared_path=spec["path"],
                )
            )
    assert original_line_hashes == transport_line_hashes
    write(
        F / "results/transport-verification.json",
        dict(
            status="passed",
            rows=sum(original_line_hashes.values()),
            every_parent_line_byte_identical=True,
            source_homogeneous_files=len(sources),
        ),
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    directories = {
        Path(p.removeprefix("presto").lstrip(".").replace(".", "/"))
        for p in config["tool"]["setuptools"]["packages"]
    }
    files = sorted(
        {str(p.relative_to(ROOT)) for d in directories for p in (ROOT / d).glob("*.py")}
    )
    files.append("data/b2m_sequences.csv")
    runtime = UPLOAD / "runtimes/shared/presto"
    for relative in files:
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    sys.path.insert(0, str(ROOT.parent))
    import torch
    from presto.models import Presto, PrestoConfig
    from presto.inference.modular import ModularPredictor  # noqa: F401
    from presto.training.modular_sampling import epoch_order

    torch.set_num_threads(1)
    pins = dict(
        source_git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_git_dirty=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        runtimes={
            "shared": dict(
                local_runtime=str(runtime), files={p: sha(runtime / p) for p in files}
            )
        },
        conditions={},
        manifests={},
        input_sources=sources,
        cohort_data_pins_sha256=sha(F / "reproduce/data-pins.json"),
        independent_verification_sha256=sha(
            F / "results/data-independent-verification.json"
        ),
        primary_ic50_files={},
        initial_common_hashes={},
        projected_exposures={},
        evaluation_scope=dp["scope"],
        census=dp["counts"],
    )
    # IC50 files can contain both original sources; worker consumes a list.
    for split in ("train", "validation", "test"):
        pins["primary_ic50_files"][split] = [
            s["path"]
            for s in sources
            if s["endpoint"] == "ic50" and s["split"] == split
        ]
    counts = Counter()
    for row in row_metadata.values():
        counts[row.endpoint] += 1
    auxiliary = sum(n for e, n in counts.items() if e != "ic50")
    weights = [
        {"values": [e], "weight": 1.0 if e == "ic50" else counts[e] / auxiliary}
        for e in ENDPOINTS
    ]
    common_hashes = {}
    for seed in (42, 43):
        for variant in VARIANTS:
            name = f"{variant}-s{seed}"
            six = "six" in variant
            shuffled = variant.endswith("shuffled")
            matched = "primary" in variant
            model = dict(
                endpoints=list(ENDPOINTS) if six else ["ic50"],
                mhc_classes=["I", "II"],
                species=["human"],
                identity_tasks=[],
                dropout=0.1,
                d_model=128 if variant.startswith("small") else 192,
                n_layers=2 if variant.startswith("small") else 3,
                n_heads=4 if variant.startswith("small") else 6,
                d_ff=512 if variant.startswith("small") else 768,
            )
            balance = (
                ["endpoint"]
                if matched
                else ["mhc_class"]
                if variant.endswith("balanced")
                else []
            )
            training = dict(
                seed=seed,
                batch_size=128 if matched else 64,
                max_steps=6000,
                max_epochs=10000,
                min_epochs=10000,
                patience=10000,
                identity_weight=0,
                learning_rate=3e-4,
                weight_decay=1e-4,
                checkpoint_every_steps=100,
                validation_every_steps=500,
                selection_endpoint="ic50",
                balance_by=balance,
                stratum_weights=weights if matched else [],
            )
            chosen = [
                s
                for s in sources
                if (six or s["endpoint"] == "ic50")
                and (
                    s["variant"]
                    == (
                        "shuffled"
                        if shuffled
                        and s["split"] == "train"
                        and s["endpoint"] != "ic50"
                        else "measured"
                    )
                )
            ]
            manifest_sources = [
                {
                    k: s[k]
                    for k in (
                        "path",
                        "sha256",
                        "source",
                        "population",
                        "label_policy",
                        "split_policy",
                    )
                }
                for s in chosen
            ]
            manifest = dict(
                schema="presto.modular.manifest.v1",
                model=model,
                training=training,
                sources=manifest_sources,
            )
            pins["conditions"][name] = dict(
                runtime="shared",
                classes=["I", "II"],
                seed=seed,
                max_steps=6000,
                variant=variant,
                balance_by=balance,
                stratum_weights=training["stratum_weights"],
                batch_size=training["batch_size"],
                synthetic_auxiliary_labels=shuffled,
            )
            pins["manifests"][name] = manifest
            torch.manual_seed(seed)
            initialized = Presto(PrestoConfig.from_dict(model))
            initial_hash = tensors_hash(initialized.state_dict())
            key = (seed, model["d_model"])
            assert common_hashes.setdefault(key, initial_hash) == initial_hash
            pins["initial_common_hashes"][name] = initial_hash
            # Simulate actual common-sampler prefixes without reading any targets.
            meta = [
                row_metadata[uid]
                for s in chosen
                if s["split"] == "train"
                for uid in transport_ids[s["path"]]
            ]
            rng, seen, steps = random.Random(seed), Counter(), 0
            while steps < 6000:
                order = epoch_order(
                    meta, range(len(meta)), balance, rng, training["stratum_weights"]
                )
                for offset in range(0, len(order), training["batch_size"]):
                    for i in order[offset : offset + training["batch_size"]]:
                        r = meta[i]
                        seen[r.endpoint + ":" + r.mhc[0].mhc_class] += 1
                    steps += 1
                    if steps == 6000:
                        break
            pins["projected_exposures"][name] = dict(seen)
            write(UPLOAD / "inputs" / f"{name}.json", manifest)
            write(F / "reproduce/manifests" / f"{name}.json", manifest)
    for destination in (UPLOAD / "source-pins.json", F / "reproduce/source-pins.json"):
        write(destination, pins)
    shutil.copytree(UPLOAD / "runtimes", F / "reproduce/source/runtimes")
    print(
        json.dumps(
            dict(
                conditions=list(pins["conditions"]),
                source_commit=pins["source_git_commit"],
                source_files=len(files),
                training_counts=counts,
                projected_exposures=pins["projected_exposures"],
                upload_bytes=sum(
                    p.stat().st_size for p in UPLOAD.rglob("*") if p.is_file()
                ),
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
