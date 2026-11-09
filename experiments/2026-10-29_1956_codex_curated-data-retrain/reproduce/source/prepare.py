"""Prepare the exact curation/recovery delta; the model and trainer remain shared."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import copy
import csv
import gzip
import hashlib
import json
import math
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time

import modal

FID = "2026-10-29_1956_codex_curated-data-retrain"
PARENT = "2026-10-28_1540_codex_temporal-multi-output"
CURATION = "2026-10-29_1653_codex_measurement-curation-recovery"
REVISION = "source-review-v2"
app = modal.App("presto-curated-data-prepare-20261029")
volume = modal.Volume.from_name("presto-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "hitlist==2.0.40",
        "torch==2.14.1",
        "numpy==2.5.4",
        "mhcseqs==2.6.13",
        "mhcgnomes==3.64.6",
        "pyyaml==6.0.3",
        "psutil==7.2.2",
        "tqdm==4.70.1",
    )
    .env({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
)
if modal.is_local():
    F = Path(__file__).resolve().parents[1]
    U = F.parents[1] / "artifacts" / FID / "upload"
    R = Path("/Users/iskander/code/presto/artifacts") / CURATION
    image = image.add_local_dir(U / "presto", "/runtime/presto")
    image = image.add_local_file(U / "runtime-pins.json", "/bundle/runtime-pins.json")
    image = image.add_local_file(U / "human-ensembl112.pep.fa.gz", "/bundle/proteome.fa.gz")
    image = image.add_local_file(F / "code/verify_recovery.py", "/bundle/verify_recovery.py")
    image = image.add_local_file(F / "results/blacklist-audit.json", "/bundle/blacklist-audit.json")
    for name in ("source-evidence.jsonl", "strazar_2023_recovered_unmodified.csv"):
        image = image.add_local_file(R / "recovered" / name, "/bundle/recovered/" + name)
    for name in ("recovery-admission.json", "curation-receipt.json"):
        image = image.add_local_file(F.parent / CURATION / "results" / name, "/bundle/" + name)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_sequence(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def weights(counts):
    family_weights = {
        "elution_observation": 0.60,
        "ic50": 0.20,
        "physical_other": 0.15,
        "qualitative_binding": 0.05,
    }

    def family(endpoint):
        return endpoint if endpoint in family_weights else "physical_other"

    totals = Counter()
    for (endpoint, _), n in counts.items():
        totals[family(endpoint)] += math.sqrt(n)
    return {
        k: family_weights[family(k[0])] * math.sqrt(n) / totals[family(k[0])]
        for k, n in counts.items()
    }


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    ephemeral_disk=524288,
    timeout=14400,
    max_containers=1,
    volumes={"/volume": volume},
)
def prepare():
    sys.path[:0] = ["/runtime", "/bundle"]
    from hitlist.proteome import ProteomeIndex
    from presto.data.measurement_curation import CensorDecision, curated_measurement
    from presto.data.modular import Observation, fit_target_scales, observation_target
    from presto.data.modular_background import LigandBackgroundSampler
    from presto.training.modular_sampling import weighted_quotas
    from verify_recovery import verify

    started = time.monotonic()
    root = Path("/runs") / FID
    out = root / "prepared"
    durable = Path("/volume") / FID / "revisions" / REVISION
    assert not durable.exists(), "Preserve previous attempts; choose a new revision"
    out.mkdir(parents=True)
    pins = read("/bundle/runtime-pins.json")
    for name, digest in pins["files"].items():
        assert sha(Path("/runtime/presto") / name) == digest
    old = Path("/volume") / PARENT / "revisions/kinetic-review-v2/prepared"
    parent = read(old / "prepared.json")
    parent_manifest = read(old / "wide-multi-s42.json")
    curation = Path("/volume") / CURATION
    curation_receipt = read("/bundle/curation-receipt.json")
    recovery_receipt = read("/bundle/recovery-admission.json")
    for name, meta in curation_receipt["summary"]["files"].items():
        assert sha(curation / "curated" / name) == meta["sha256"]
    local_recovery = root / "recovery-verification"
    shutil.copytree("/bundle/recovered", local_recovery / "recovered")
    shutil.copytree(curation / "recovered-admission", local_recovery / "recovered-admission")
    recovery_verification = verify(local_recovery, recovery_receipt)
    write(out / "recovery-verification.json", recovery_verification)
    with gzip.open(curation / "curated/decisions.jsonl.gz", "rt") as stream:
        decisions = {
            d.assay_iri: d for line in stream for d in [CensorDecision(**json.loads(line))]
        }
    new_physical = out / "hitlist-v4-iedb-train.jsonl"
    with (
        gzip.open(curation / "curated/train.jsonl.gz", "rb") as src,
        new_physical.open("xb") as dst,
    ):
        shutil.copyfileobj(src, dst)

    # Independently prove the exact changed/removed row set and original field equality.
    removed, changed = [], Counter()
    old_count, new_count = Counter(), Counter()

    def key(row):
        return ":".join(
            (row["split"], row["endpoint"], row["mhc"][0]["mhc_class"], row["qualifier"])
        )

    with (old / new_physical.name).open() as before, new_physical.open() as after:
        for line in before:
            original = json.loads(line)
            old_count[key(original)] += 1
            raw = original["provenance"]["raw_record"]
            decision = decisions.get(raw["assay_iri"])
            if decision is not None and decision.status == "needs_review":
                removed.append(original["observation_id"])
                continue
            row = json.loads(next(after))
            assert row["observation_id"] == original["observation_id"]
            assert row["provenance"]["raw_record"] == raw
            for field in set(original) - {"qualifier", "provenance"}:
                assert row[field] == original[field], field
            for field, value in original["provenance"].items():
                if field != "missing_qualifier_convention":
                    assert row["provenance"][field] == value, field
            if row["endpoint"] != "qualitative_binding":
                measurement = curated_measurement(raw, decision)
                assert row["qualifier"] == measurement["effective_qualifier"]
                assert row["provenance"]["measurement_curation"] == measurement
            else:
                assert row["qualifier"] == original["qualifier"]
            if row["qualifier"] != original["qualifier"]:
                changed[row["qualifier"]] += 1
            new_count[key(row)] += 1
        assert not after.read()
    assert len(removed) == 618 and changed == {">=": 482, "<=": 24}
    assert sum(new_count.values()) == 233719
    print("Verified physical curation delta", dict(changed), len(removed), flush=True)

    audit = read("/bundle/blacklist-audit.json")
    assert audit["status"] == "audited"
    with open("/bundle/recovered/strazar_2023_recovered_unmodified.csv") as stream:
        reviewed_peptides = {r["peptide"] for r in csv.DictReader(stream)}
    assert len(reviewed_peptides) == audit["reviewed_unique_sequences"] == 11100
    expected_removed_backgrounds = {
        r["observation_id"]: r
        for r in audit["intersections"]
        if r["generated"] and r["split"] == "train"
    }
    assert len(expected_removed_backgrounds) == 35
    removed_backgrounds, removed_background_counts, removed_populations = [], Counter(), Counter()
    sources, input_receipts, heldout = [], {}, set()
    for spec in parent_manifest["sources"]:
        name = Path(spec["path"]).name
        assert sha(old / name) == spec["sha256"]
        path = out / name
        if name == "hitlist_ligand_background-train.jsonl":
            assert spec["sha256"] == audit["source_hashes"][name]
            with (old / name).open("rb") as src, path.open("xb") as dst:
                for line in src:
                    row = json.loads(line)
                    if row["peptide"] in reviewed_peptides:
                        expected = expected_removed_backgrounds[row["observation_id"]]
                        assert row["split"] == "train" and row["value"] == 0
                        assert sha_sequence(row["peptide"]) == expected["peptide_sha256"]
                        removed_backgrounds.append(row["observation_id"])
                        removed_background_counts[key(row)] += 1
                        removed_populations[row["provenance"]["evaluation_population"]] += 1
                    else:
                        dst.write(line)
            assert set(removed_backgrounds) == set(expected_removed_backgrounds)
        elif path != new_physical:
            shutil.copyfile(old / name, path)
            assert sha(path) == spec["sha256"]
        if name.endswith(("-test.jsonl", "-validation.jsonl")):
            with path.open() as stream:
                heldout.update(json.loads(line)["peptide"] for line in stream)
        new_spec = dict(spec, path=str(path), sha256=sha(path))
        if path == new_physical:
            new_spec["label_policy"] += (
                "; source-scoped inferred bounds with raw provenance; unresolved heaps withheld"
            )
        elif name == "hitlist_ligand_background-train.jsonl":
            new_spec["label_policy"] += "; 35 newly source-verified positive sequences excluded"
        sources.append(new_spec)
        input_receipts[name] = dict(before_sha256=spec["sha256"], after_sha256=new_spec["sha256"])

    db = sqlite3.connect("file:/volume/" + PARENT + "/admission/admitted.sqlite?mode=ro", uri=True)
    positives = {p for (p,) in db.execute("SELECT peptide FROM positive_peptides")}
    new_blacklist_sequences = len(reviewed_peptides - positives)
    assert new_blacklist_sequences == audit["missing_reviewed_from_old_blacklist"] == 292
    positives.update(reviewed_peptides)
    with gzip.open(curation / "recovered-admission/train.jsonl.gz", "rt") as stream:
        recovered = [Observation.from_dict(json.loads(line)) for line in stream]
    assert len(recovered) == 13550
    extra_embargo = [r.observation_id for r in recovered if r.peptide in heldout]
    assert len(extra_embargo) == audit["new_heldout_embargo_observations"] == 1
    recovered = [r for r in recovered if r.peptide not in heldout]
    assert len(recovered) == 13549
    recovered_peptides = {r.peptide for r in recovered}
    assert recovered_peptides <= positives, (
        "New positive outside frozen blacklist; review old backgrounds"
    )
    assert not recovered_peptides & heldout
    for start in range(0, len(recovered), 400):
        ids = [r.observation_id for r in recovered[start : start + 400]]
        duplicates = db.execute(
            "SELECT id FROM rows WHERE excluded=0 AND id IN (" + ",".join("?" * len(ids)) + ")", ids
        ).fetchall()
        assert not duplicates
    db.close()
    assert sha("/bundle/proteome.fa.gz") == pins["proteome"]["sha256"]
    fasta = root / "proteome.fa"
    with gzip.open("/bundle/proteome.fa.gz", "rb") as src, fasta.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    proteome = ProteomeIndex.from_fasta(str(fasta))
    splits = {p: "validation_or_test" for p in heldout}
    sampler = LigandBackgroundSampler(
        proteome,
        seed=20261028,
        reference=PARENT + ":length-matched-background-v1",
        positive_peptides=positives,
        peptide_splits=splits,
    )
    added_counts, background_counts = Counter(), Counter()
    positive_path, background_path = (
        out / "recovered-ms-train.jsonl",
        out / "recovered-background-train.jsonl",
    )
    with positive_path.open("x") as pstream, background_path.open("x") as bstream:
        for row in sorted(recovered, key=lambda r: r.observation_id):
            row = replace(
                row,
                provenance={
                    **row.provenance,
                    "study_id": "pmid:37301199",
                    "group_aliases": ["pmid:37301199"],
                    "study_component_years": [2023],
                    "evaluation_era": "older_or_mixed",
                    "partition_policy": FID + ":fixed-parent-holdouts-v1",
                },
            )
            bg = sampler.sample(row, replicate=0)
            assert bg.peptide not in positives and bg.peptide not in heldout
            assert len(bg.peptide) == len(row.peptide) and bg.mhc == row.mhc
            population = "synthetic_" + bg.provenance["background_kind"]
            for value, stream in ((row, pstream), (bg, bstream)):
                value = replace(
                    value,
                    provenance={
                        **value.provenance,
                        "evaluation_population": population,
                        "evaluation_era": "older_or_mixed",
                    },
                )
                value.validate()
                stream.write(
                    json.dumps(value.to_dict(), separators=(",", ":"), allow_nan=False) + "\n"
                )
                added_counts[key(value.to_dict())] += 1
            background_counts[population] += 1
    assert sum(background_counts.values()) == 13549
    for path, source in (
        (positive_path, "hitlist_native_ms"),
        (background_path, "hitlist_ligand_background"),
    ):
        sources.append(
            dict(
                path=str(path),
                sha256=sha(path),
                source=source,
                population=(
                    "Source-reviewed unmodified human monoallelic MHC-II ligands from "
                    "PMID37301199 and declared matched unobserved comparisons"
                ),
                label_policy=(
                    "Source-backed positive MS forms; one generated unobserved background "
                    "per TRAIN positive; no measured nonbinder claim"
                ),
                split_policy=(
                    "New older TRAIN study; parent heldout peptide embargo; "
                    "source-extended frozen known-positive blacklist"
                ),
            )
        )
    del proteome, positives, splits, heldout, recovered, reviewed_peptides

    p = copy.deepcopy(parent)
    p.update(family=FID, status="prepared")
    p["counts"] = dict(
        Counter(p["counts"]) - old_count + new_count + added_counts - removed_background_counts
    )
    strata = Counter()
    for k, n in p["counts"].items():
        split, endpoint, cls, _ = k.split(":")
        if split == "train":
            strata[endpoint, cls] += n
    sampling = weights(strata)
    quotas = weighted_quotas(strata, sampling)
    p["train_strata"] = {":".join(k): n for k, n in sorted(strata.items())}
    p["epoch_quotas"] = {":".join(k): n for k, n in sorted(quotas.items())}
    p["weighted_epoch_examples"] = sum(quotas.values())
    p["steps"] = math.ceil(sum(quotas.values()) / 128)
    exact = {}

    def physical():
        with new_physical.open() as stream:
            for line in stream:
                row = Observation.from_dict(json.loads(line))
                if row.qualifier == "=" and row.endpoint != "qualitative_binding":
                    exact.setdefault(row.endpoint, []).append(observation_target(row))
                yield row

    scales = fit_target_scales(physical())
    for endpoint, scale in scales.items():
        assert scale.exact_count == len(exact[endpoint])
        assert math.isclose(scale.mean, statistics.fmean(exact[endpoint]), abs_tol=1e-10)
        assert math.isclose(
            scale.scale, max(scale.floor, statistics.pstdev(exact[endpoint])), abs_tol=1e-10
        )
    p["expected_scales"] = {k: vars(v) for k, v in scales.items()}
    p["manifests"] = {}
    for seed in (42, 43):
        name = "curated-multi-s" + str(seed)
        manifest = copy.deepcopy(parent_manifest)
        manifest["sources"] = sources
        manifest["training"].update(
            seed=seed,
            max_steps=p["steps"],
            max_epochs=1,
            stratum_weights=[dict(values=list(k), weight=v) for k, v in sorted(sampling.items())],
        )
        path = out / (name + ".json")
        write(path, manifest)
        p["manifests"][name] = dict(
            path=str(path), sha256=sha(path), seed=seed, max_steps=p["steps"]
        )
    p["source_review"] = dict(
        revision=REVISION,
        curation_family=CURATION,
        removed_rows=618,
        inferred_bounds=dict(changed),
        recovered_positive_rows=13549,
        added_background_rows=13549,
        removed_generated_training_rows=35,
        additional_recovered_holdout_embargo=1,
        new_blacklist_sequences=new_blacklist_sequences,
        suspect_generated_test_rows=3,
        blacklist_audit_sha256=sha("/bundle/blacklist-audit.json"),
        recovered_positive_blacklist_verified=True,
        recovered_holdout_embargo_verified=True,
        recovery_verification=recovery_verification,
        parent_family=PARENT,
        previous_test_inspection=True,
        new_background_populations=dict(background_counts),
    )
    p["verification"] = dict(
        status="passed",
        original_nonphysical_sources_unchanged_except_declared_background_removals=True,
        validation_test_bytes_unchanged=True,
        exact_physical_delta_verified=True,
        recovered_source_forms_verified=True,
        known_positive_and_heldout_exclusions_verified=True,
        canonical_preflight="pending_in_trainers",
        input_receipts=input_receipts,
    )
    p["background_contract"]["new_recovered_backgrounds"] = dict(background_counts)
    p["dispositions"]["train:inferred_censor_review_withheld"] = 618
    for cohort, sign in ((old_count, -1), (new_count, 1)):
        for k, n in cohort.items():
            split, endpoint, _, _ = k.split(":")
            p["physical_observations"][split + ":" + endpoint] += sign * n
    p["physical_observations"]["train:elution_observation"] += 13549
    p["dispositions"]["train:recovered_positive_background_conflicts"] = 35
    p["dispositions"]["train:additional_recovered_heldout_embargo"] = 1
    for population, n in removed_populations.items():
        p["background_counts"]["train:" + population] -= n
    for population, n in background_counts.items():
        target = "train:" + population
        p["background_counts"][target] = p["background_counts"].get(target, 0) + n
    p["preparation_seconds"] = time.monotonic() - started
    write(out / "prepared.json", p)
    write(out / "removed-observation-ids.json", removed)
    write(out / "removed-background-observation-ids.json", removed_backgrounds)
    write(out / "additional-embargo-observation-ids.json", extra_embargo)
    shutil.copyfile("/bundle/blacklist-audit.json", out / "blacklist-audit.json")
    shutil.copyfile(old / "study-selection.json", out / "study-selection.json")
    selection = read(old / "study-selection.json")
    assert not any("pmid:37301199" in c["aliases"] for c in selection["components"])
    write(
        out / "additional-study-selection.json",
        dict(
            group_id="pmid:37301199",
            split="train",
            year=2023,
            source_family=CURATION,
            heldout_peptide_embargo="all original validation/test sequences, including backgrounds",
            original_selection_sha256=sha(out / "study-selection.json"),
        ),
    )
    durable.mkdir(parents=True)
    shutil.copytree(out, durable / "prepared")
    p["prepared_record_sha256"] = sha(out / "prepared.json")
    p["review_metadata"] = {Path(s["path"]).name: read(s["path"]) for s in p["manifests"].values()}
    p["review_metadata"]["preflight.json"] = dict(
        status="pending_in_trainers", expected_scales=p["expected_scales"]
    )
    write(durable / "preparation-result.json", p)
    volume.commit()
    print(
        "Prepared",
        sum(strata.values()),
        "TRAIN observations;",
        p["steps"],
        "updates/seed",
        flush=True,
    )
    return p


@app.local_entrypoint()
def main():
    f = Path(__file__).resolve().parents[1]
    assert not (f / "reproduce/prepare-call.json").exists()
    for name in ("prepare.py", "verify_recovery.py"):
        dest = f / "reproduce/source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f / "code" / name, dest)
    write(
        f / "reproduce/prepare-launch.json",
        dict(
            argv=[sys.executable, "-m", "modal", "run", "--detach", __file__],
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            git_dirty=subprocess.check_output(["git", "status", "--short"], text=True),
            source_sha256=sha(__file__),
            runtime_pins_sha256=sha(f / "reproduce/runtime-pins.json"),
            started_unix=time.time(),
            requested_gpu=None,
            cpu=8,
            memory_mib=65536,
        ),
    )
    call = prepare.spawn()
    write(f / "reproduce/prepare-call.json", dict(call_id=call.object_id))
    result = call.get()
    write(f / "results/prepared.json", result)
    print(
        json.dumps(
            dict(status=result["status"], steps=result["steps"], counts=result["source_review"])
        ),
        flush=True,
    )
