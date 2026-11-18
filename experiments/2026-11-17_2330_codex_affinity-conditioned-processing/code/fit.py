"""Registered continuation recipes around the canonical manifest trainer."""
import gzip
import hashlib
import json
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from common import BROAD,NAME,rows,sha,dump
PARENT_RUN=Path('/volume')/BROAD/'fits/expanded-state-sqrt-s42-n6000'
PARENT=PARENT_RUN/'raw/selected.pt'
PARENT_SHA='1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68'
ARMS={'continued','paired','paired-junction'}


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


def compile_manifest(root,out,arm,seed,steps):
    from presto.training.modular import load_checkpoint
    assert arm in ARMS and seed in (42,43) and steps in (16,2000)
    assert sha(PARENT)==PARENT_SHA
    inherited=json.loads((PARENT_RUN/'manifest.json').read_text())
    model,_,_=load_checkpoint(PARENT,device='cpu')
    config=model.config.to_dict();del model
    if arm=='paired-junction':config['elution_junction_processing']=True
    sources=inherited['sources']
    for source in sources:assert sha(source['path'])==source['sha256']
    training={**inherited['training'],'seed':seed,'max_steps':steps,'learning_rate':3e-5,
              'checkpoint_every_steps':min(500,steps),'validation_every_steps':min(500,steps),
              'lr_schedule':{'warmup_steps':min(100,max(1,steps//10)),'decay':'cosine','decay_steps':steps,'min_lr_fraction':.1},
              'no_test_reason':'Affinity-conditioned processing development; canonical TEST outcomes remain sealed'}
    if arm!='continued':
        receipt=json.loads((root/'processing-preparation.json').read_text());assert receipt['status']=='prepared'
        source=next(x for x in receipt['sources'] if x['path'].endswith('processing-backgrounds-train.jsonl'))
        assert sha(source['path'])==source['sha256']
        sources=[*sources,{'path':source['path'],'sha256':source['sha256'],'source':'hitlist_native_ms',
            'population':'Auxiliary protected source backgrounds referring to original native positive IDs',
            'label_policy':'Observed-positive versus unobserved generated background; conditional BA balance, not predictor-defined negatives',
            'split_policy':'TRAIN only; exact same protected sequence identities, protein groups and conditions; excluded from ordinary replay'}]
        training['processing_pair_training']={'weight':.5,'pairs_per_class':4}
    if steps==16:
        training.update(coverage_policy='diagnostic',coverage_exception_reason='16-update full-data integration; no predictive gain claim')
    manifest={'schema':'presto.modular.manifest.v1','model':config,'training':training,'sources':sources,
        'pretrained':{'path':str(PARENT),'sha256':PARENT_SHA,'freeze':[],
                      **({'allow_junction_expansion':True} if arm=='paired-junction' else {})}}
    out.mkdir(parents=True,exist_ok=True);dump(out/'manifest.json',manifest)
    dump(out/'inherited-recipe.json',{'parent_manifest_sha256':sha(PARENT_RUN/'manifest.json'),
        'ordinary_sources_and_weights_unchanged':True,'auxiliary_training':training.get('processing_pair_training'),
        'selected_primary_endpoint':'equal class mean of equal-study presentation losses'})
    return manifest


def resume_sources(out):
    """Restore worker-local bytes without rewriting the original run contract."""
    receipt = json.loads((out / "source-staging.json").read_text())
    assert sha(out / "manifest.json") == receipt["original_manifest_sha256"]
    execution = out / "execution-manifest.json"
    assert sha(execution) == receipt["execution_manifest_sha256"]
    for source in receipt["sources"]:
        original, local = Path(source["original_path"]), Path(source["local_path"])
        assert sha(original) == source["sha256"]
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists():
            raise ValueError("resume staging destination already exists: " + str(local))
        shutil.copyfile(original, local)
        assert sha(local) == source["sha256"] and local.stat().st_size == source["bytes"]
    return execution


def fit(root, out, arm, seed, steps, commit=lambda: None, resume_run=False):
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular import load_checkpoint
    from presto.training.modular_run import fit_index

    start = time.monotonic()
    manifest = (
        json.loads((out / "manifest.json").read_text())
        if resume_run
        else compile_manifest(root, out, arm, seed, steps)
    )
    if resume_run:
        assert manifest["training"]["seed"] == seed and manifest["training"]["max_steps"] == steps
    manifest_seconds = time.monotonic() - start
    print(json.dumps({"stage": "manifest_ready", "seconds": manifest_seconds}), flush=True)
    commit()
    stage_start = time.monotonic()
    execution_path = resume_sources(out) if resume_run else stage_sources(manifest, out)
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
        summary = fit_index(index, out / "raw", device="cuda", resume=resume_run)
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
        i: (row.endpoint, next(iter(row.mhc_classes)), row.study) for i, row in metadata.items()
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
            "ordered_source_indices_sha256": hashlib.sha256(
                json.dumps(visited, separators=(",", ":")).encode()
            ).hexdigest(),
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
    (out / "actual-training-exposure.json").write_text(json.dumps(exposure, indent=2) + "\n")
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
            "total_wall_seconds": (
                time.time() - (out / "hardware.json").stat().st_mtime
                if resume_run
                else time.monotonic() - start
            ),
            **(
                {
                    "resumed_segment_wall_seconds": time.monotonic() - start,
                    "wall_time_scope": (
                        "from original hardware receipt to completion, "
                        "including interruption and recovery"
                    ),
                }
                if resume_run
                else {}
            ),
        },
        "resumed_from_durable_state": resume_run,
        "coverage": summary["coverage_gate"]["status"],
        "checkpoint_sha256": sha(out / "raw/selected.pt"),
        "path": str(out),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    commit()
    return receipt
