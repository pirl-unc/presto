"""Experiment recipes around the canonical manifest trainer; no alternate optimizer."""

import gzip
import json
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

from prepare import PRIOR, rows, sha

PARENT = Path(
    "/volume/2026-11-08_1800_codex_unified-parity-retrain/fits-v3/reference/raw/selected.pt"
)
PARENT_SHA = "1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a"
ARMS = {
    "pilot-balanced",
    "expanded-none",
    "expanded-state",
    "expanded-ii-transfer",
    "expanded-pathway-state",
}


def training_context(paths):
    """Vocabulary from exactly the observations admitted to this arm's TRAIN."""
    values = {
        name: set()
        for name in ("cell_types", "cytokines", "interventions", "processing_pathways")
    }
    for path in paths:
        for row in rows(path):
            if row["split"] != "train":
                raise ValueError("context vocabulary source must contain TRAIN only")
            context = row["context"]
            cell = context.get("cell_type_override") or context.get("cell_type")
            if cell is not None:
                values["cell_types"].add(cell)
            for family, override in (
                ("cytokines", "cytokine_overrides"),
                ("interventions", "intervention_overrides"),
                ("processing_pathways", "pathway_overrides"),
            ):
                values[family].update(context.get(family, {}))
                values[family].update(context.get(override, {}))
    return {name: sorted(items) for name, items in values.items()}


def stage_sources(manifest, out, local_root=Path("/tmp/presto-fit-inputs")):
    """Keep canonical per-row integrity checks on verified worker-local bytes."""
    start = time.monotonic()
    folder = local_root / out.name
    folder.mkdir(parents=True, exist_ok=False)
    sources, receipt = [], []
    for i, source in enumerate(manifest["sources"]):
        original = Path(source["path"])
        local = folder / f"{i:02d}-{original.name}"
        shutil.copyfile(original, local)
        if sha(local) != source["sha256"]:
            raise ValueError("staged source hash mismatch: " + str(original))
        sources.append({**source, "path": str(local)})
        receipt.append(
            {
                "original_path": str(original),
                "local_path": str(local),
                "sha256": source["sha256"],
                "bytes": local.stat().st_size,
            }
        )
    execution = {**manifest, "sources": sources}
    path = out / "execution-manifest.json"
    path.write_text(json.dumps(execution, indent=2) + "\n")
    (out / "source-staging.json").write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - start,
                "sources": receipt,
                "original_manifest_sha256": sha(out / "manifest.json"),
                "execution_manifest_sha256": sha(path),
                "policy": "byte-identical verified copies; unchanged canonical integrity checks",
            },
            indent=2,
        )
        + "\n"
    )
    return path


def compile_manifest(root, out, arm, seed, steps):
    from presto.training.modular import load_checkpoint
    from presto.training.modular_controls import study_of
    from project_context import Projection

    if arm not in ARMS:
        raise ValueError("unregistered arm")
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("parent checkpoint changed")
    report = json.loads((root / "preparation.json").read_text())
    if report["status"] != "prepared":
        raise ValueError("preparation incomplete")
    audit = json.loads((root / "audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("independent union audit required")
    d = root / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    parent, _, _ = load_checkpoint(PARENT, device="cpu")
    config = parent.config.to_dict()
    del parent
    has_state = arm != "expanded-none"
    if has_state:
        if arm == "expanded-ii-transfer":
            paths = [
                d / name
                for name in (
                    "expanded-II-train.jsonl",
                    "original-apc-I-train.jsonl",
                    "original-broad-I-train.jsonl",
                )
            ]
            for path in paths:
                if sha(path) != report["files"][path.name]["sha256"]:
                    raise ValueError("transfer TRAIN source changed")
            ctx = training_context(paths)
        else:
            ctx = (
                report["model_context"]
                if arm != "pilot-balanced"
                else json.loads((PRIOR / "inputs/apc-cohort.json").read_text())[
                    "model_context"
                ]
            )
        config.update(ctx, state_bottleneck=64, state_mode="harmonized")
    config["elution_detach_affinity"] = False
    if arm == "expanded-pathway-state":
        config["elution_pathway_conditioning"] = True
    names = [
        "affinity-train.jsonl",
        "affinity-validation.jsonl",
        "apc-validation.jsonl",
        "broad-validation.jsonl",
    ]
    if report["files"]["expanded-validation.jsonl"]["rows"]:
        names += ["expanded-validation.jsonl"]
    if arm == "pilot-balanced":
        names += ["apc-train.jsonl", "broad-train.jsonl"]
    elif arm == "expanded-ii-transfer":
        names += [
            "expanded-II-train.jsonl",
            "original-apc-I-train.jsonl",
            "original-broad-I-train.jsonl",
        ]
    else:
        names += ["expanded-train.jsonl"]
    sources = []
    projections = {}
    strata = Counter()
    for name in names:
        p = d / name
        if sha(p) != report["files"][name]["sha256"]:
            raise ValueError("prepared source changed: " + name)
        original_sha = sha(p)
        if "validation" in name:
            projection = Projection(config)
            projected_path = out / "validation-inputs" / name
            projected_path.parent.mkdir(parents=True, exist_ok=True)
            with projected_path.open("w", buffering=1024 * 1024) as f:
                for r in rows(p):
                    _, encoded = projection.record(r)
                    f.write(encoded)
            receipt = projection.receipt()
            assert sha(projected_path) == receipt["projected_jsonl_sha256"]
            projections[name] = {"original_sha256": original_sha, **receipt}
            p = projected_path
        sources.append(
            {
                "path": str(p),
                "sha256": sha(p),
                "source": next(rows(p))["source"],
                "population": "Measured concentration replay and source-reported presentation with declared sampled backgrounds",
                "label_policy": "Literal assay/qualifier; MS detection vs unobserved same-possible-source background, unknown APC preserved; no predictor-derived labels",
                "split_policy": "Frozen validation; protected sequence families; full uncapped native admission with source-family split and historical TRAIN exposure exclusions",
            }
        )
        if "train" in name:
            for r in rows(p):
                study = study_of(r["provenance"])
                if study is None:
                    raise ValueError("missing training study provenance")
                classes = {m.get("class_override") or m["mhc_class"] for m in r["mhc"]}
                if len(classes) != 1:
                    raise ValueError("ambiguous training class")
                strata[(r["endpoint"], next(iter(classes)), str(study))] += 1
    binding = sum(n for k, n in strata.items() if k[0] != "elution_observation")
    studies = {
        cls: {k[2] for k in strata if k[:2] == ("elution_observation", cls)}
        for cls in ("I", "II")
    }
    if not binding or any(not v for v in studies.values()):
        raise ValueError("all three task families required")
    weights = [
        {
            "values": list(k),
            "weight": 0.5 * n / binding
            if k[0] != "elution_observation"
            else 0.25 / len(studies[k[1]]),
        }
        for k, n in sorted(strata.items())
    ]
    training = {
        "seed": seed,
        "batch_size": 64,
        "max_steps": steps,
        "max_epochs": 100000,
        "min_epochs": 1,
        "patience": 100000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "identity_weight": 0.0,
        "balance_by": ["endpoint", "mhc_class", "study"],
        "stratum_weights": weights,
        "sampling_mode": "weighted_steps",
        "lr_schedule": {
            "warmup_steps": min(100, max(1, steps // 10)),
            "decay": "cosine",
            "decay_steps": steps,
            "min_lr_fraction": 0.1,
        },
        "checkpoint_every_steps": min(1000, steps),
        "validation_every_steps": min(1000, steps),
        "selection_endpoint": "elution_observation",
        "selection_reduction": "class_study_mean",
        "train_evaluation_max_rows": 512,
        "test_policy": "no_test_training_diagnostic",
        "no_test_reason": "Shared presentation development; canonical TEST remains sealed",
        "context_support_policy": "predictive",
        "coverage_policy": "required",
    }
    if steps < 1000:
        training.update(
            coverage_policy="diagnostic",
            coverage_exception_reason="Short integration preflight; no predictive gain claim",
        )
    manifest = {
        "schema": "presto.modular.manifest.v1",
        "model": config,
        "sources": sources,
        "pretrained": {
            "path": str(PARENT),
            "sha256": PARENT_SHA,
            "freeze": [],
            **({"allow_apc_expansion": True} if has_state else {}),
            **(
                {"allow_pathway_expansion": True}
                if arm == "expanded-pathway-state"
                else {}
            ),
        },
        "training": training,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "validation-context-projection.json").write_text(
        json.dumps(projections, indent=2) + "\n"
    )
    (out / "strata.json").write_text(
        json.dumps(
            [{"values": list(k), "rows": n} for k, n in sorted(strata.items())],
            indent=2,
        )
        + "\n"
    )
    return manifest


def fit(root, out, arm, seed, steps, commit=lambda: None):
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular import load_checkpoint
    from presto.training.modular_run import fit_index

    start = time.monotonic()
    manifest = compile_manifest(root, out, arm, seed, steps)
    manifest_seconds = time.monotonic() - start
    print(
        json.dumps({"stage": "manifest_ready", "seconds": manifest_seconds}), flush=True
    )
    commit()
    stage_start = time.monotonic()
    execution_path = stage_sources(manifest, out)
    stage_seconds = time.monotonic() - stage_start
    print(
        json.dumps({"stage": "local_sources_verified", "seconds": stage_seconds}),
        flush=True,
    )
    index_start = time.monotonic()
    index = ObservationIndex(execution_path)
    index_seconds = time.monotonic() - index_start
    print(
        json.dumps({"stage": "admission_index_ready", "seconds": index_seconds}),
        flush=True,
    )
    stop = threading.Event()

    def publish():
        while not stop.wait(45):
            try:
                commit()
            except Exception as error:  # noqa: BLE001 - final mandatory commit still raises
                print("Periodic commit retry:", repr(error), flush=True)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    trainer_start = time.monotonic()
    try:
        summary = fit_index(index, out / "raw", device="cuda")
    finally:
        stop.set()
        thread.join(timeout=2)
        commit()
    trainer_seconds = time.monotonic() - trainer_start
    model, _, selected_payload = load_checkpoint(out / "raw/selected.pt", device="cuda")
    parent, _, _ = load_checkpoint(PARENT, device="cuda")
    old = dict(parent.named_parameters())
    changed = {
        n: not torch.equal(p, old[n])
        for n, p in model.named_parameters()
        if n in old and p.shape == old[n].shape
    }
    updates = {
        "frozen_parameters": list(model.frozen_parameters),
        "shared_parameters_changed": changed,
        "all_shared_parameters_changed": all(changed.values()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (out / "update-check.json").write_text(json.dumps(updates, indent=2) + "\n")
    # Eligible corpus size differs from examples actually visited by a finite run.
    resume = torch.load(out / "raw/resume.pt", map_location="cpu", weights_only=True)
    state = resume["metadata"]["resume"]["state"]
    order = state["order"][: state["position"]]
    batch_size = index.manifest["training"]["batch_size"]
    assert len(order) == summary["steps"] * batch_size
    selected_steps = selected_payload["metadata"]["steps"]
    metadata = {i: index.metadata(i) for i in set(order)}
    strata_by_index = {
        i: (row.endpoint, next(iter(row.mhc_classes)), row.study)
        for i, row in metadata.items()
    }
    exposure = {"selected_steps": selected_steps, "executed_steps": summary["steps"]}
    for scope, visited in [
        ("selected", order[: selected_steps * batch_size]),
        ("executed", order),
    ]:
        counts = Counter(strata_by_index[i] for i in visited)
        unique = Counter(strata_by_index[i] for i in set(visited))
        exposure[scope] = {
            "draws": len(visited),
            "unique_rows": len(set(visited)),
            "strata": [
                {
                    "endpoint": ep,
                    "class": cls,
                    "study": study,
                    "draws": n,
                    "unique_rows": unique[(ep, cls, study)],
                }
                for (ep, cls, study), n in sorted(counts.items())
            ],
        }
    (out / "actual-training-exposure.json").write_text(
        json.dumps(exposure, indent=2) + "\n"
    )
    del resume, metadata
    # Preserve decoded prediction hashes while storing compact artifacts for collection.
    for p in (out / "raw").glob("*predictions.jsonl"):
        digest = sha(p)
        with (
            p.open("rb") as source,
            gzip.open(str(p) + ".gz", "wb", compresslevel=6) as target,
        ):
            shutil.copyfileobj(source, target)
        (out / (p.name + ".sha256")).write_text(digest + "\n")
        # Keep canonical raw artifacts remotely; local collection requests gzip only.
    receipt = {
        "arm": arm,
        "seed": seed,
        "steps": summary["steps"],
        "seconds": summary["elapsed_seconds"],
        "runtime": {
            "manifest_seconds": manifest_seconds,
            "admission_index_seconds": index_seconds,
            "local_staging_seconds": stage_seconds,
            "canonical_trainer_including_preflight_seconds": trainer_seconds,
            "training_and_evaluation_seconds": summary["elapsed_seconds"],
            "total_wall_seconds": time.monotonic() - start,
        },
        "coverage": summary["coverage_gate"]["status"],
        "checkpoint_sha256": sha(out / "raw/selected.pt"),
        "path": str(out),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    commit()
    return receipt
