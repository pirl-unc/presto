"""Fixed MHC species controls and canonical identity-only fits."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch

from common import CONDITIONS, FAMILY, ORIGINAL_SHA, RAW, ROOT, guarded, receipt, write
from presto.data.modular import AA
from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular_resume import SOURCE_FILES
from presto.training.modular_run import fit_index, preflight

SPECIES = ("human", "mouse", "rat", "rhesus_macaque")
SOURCE = Path(
    "/Users/iskander/code/presto/.worktrees/presto-modular-model/data/modular_cohorts/2026-10-21-mhc-reference/manifest.json"
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    if (FAMILY / "reproduce/launch.json").exists():
        raise FileExistsError("existing family must be preserved")
    if file_sha256(SOURCE) != ORIGINAL_SHA:
        raise ValueError("qualified original manifest changed")
    original = read(SOURCE)
    for source in original["sources"]:
        source["path"] = str((SOURCE.parent / source["path"]).resolve())
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError("qualified observations changed")
    pins = {}
    for condition, seed in CONDITIONS.items():
        manifest = json.loads(json.dumps(original))
        manifest["model"].update(dropout=0.1, max_length=512)
        manifest["training"] = dict(
            seed=seed,
            batch_size=64,
            max_epochs=50,
            min_epochs=10,
            patience=10,
            max_steps=None,
            learning_rate=3e-4,
            weight_decay=1e-4,
            identity_weight=1,
            identity_batch_size=64,
            coverage_policy="required",
            test_policy="evaluate_selected_once",
            checkpoint_every_steps=50,
        )
        path = FAMILY / "reproduce" / f"{condition}-manifest.json"
        write(path, manifest)
        pins[condition] = {"path": str(path), "sha256": file_sha256(path)}
    RAW.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        target = FAMILY / "reproduce/source/runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for path in (FAMILY / "code").glob("*.py"):
        shutil.copyfile(path, FAMILY / "reproduce/source" / path.name)
    state = receipt()
    state.update(
        original_manifest_path=str(SOURCE),
        original_manifest_sha256=ORIGINAL_SHA,
        manifests=pins,
        raw_artifacts=str(RAW),
        species=list(SPECIES),
    )
    write(FAMILY / "reproduce/launch.json", state)
    print("Prepared immutable species conditions", flush=True)


def index_for(condition):
    launch = read(FAMILY / "reproduce/launch.json")
    for name, digest in launch["runtime_hashes"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("runtime changed: " + name)
    for name, digest in launch["code_hashes"].items():
        if file_sha256(FAMILY / "code" / name) != digest:
            raise ValueError("family source changed: " + name)
    pin = launch["manifests"][condition]
    if file_sha256(pin["path"]) != pin["sha256"]:
        raise ValueError("manifest changed")
    return ObservationIndex(pin["path"])


def features(rows):
    result = []
    for row in rows:
        if len(row.identity_labels) != 1 or len(row.mhc) != 1:
            raise ValueError("expected one domain annotation")
        key, species = next(iter(row.identity_labels.items()))
        if key not in ("mhc1", "mhc2") or species not in SPECIES:
            raise ValueError("unexpected annotation")
        first, second = row.mhc[0].groove1, row.mhc[0].groove2
        if bool(first) == bool(second) or bool(second) != (key == "mhc2"):
            raise ValueError("domain slot differs from target")
        seq = first or second
        result.append([len(seq), *[seq.count(a) / len(seq) for a in AA], float(bool(second))])
    return np.asarray(result, float)


def baselines():
    index = index_for("seed42")
    _, _, _, admission = preflight(index)
    destination = FAMILY / "results/baselines"
    destination.mkdir(exist_ok=False)
    rows = {
        split: [index[i] for i in index.partitions[f"{split}:identity"]]
        for split in ("train", "validation", "test")
    }
    labels = {
        s: np.asarray([SPECIES.index(next(iter(r.identity_labels.values()))) for r in rs])
        for s, rs in rows.items()
    }
    x = features(rows["train"])
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x - center) / scale])
    y = np.eye(4)[labels["train"]]

    def objective(flat):
        w = flat.reshape(z.shape[1], 4)
        logits = z @ w
        loss = np.sum(logsumexp(logits, axis=1) - np.sum(y * logits, axis=1))
        penalty = w.copy()
        penalty[0] = 0
        return loss + 0.5 * np.sum(penalty**2), (
            z.T @ (softmax(logits, axis=1) - y) + penalty
        ).ravel()

    fit = minimize(
        objective,
        np.zeros(z.shape[1] * 4),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "gtol": 1e-7, "ftol": 1e-15, "maxls": 50},
    )
    gradient = float(np.max(np.abs(objective(fit.x)[1])))
    parameters = dict(
        features=["length", *list(AA), "groove2"],
        center=center.tolist(),
        scale=scale.tolist(),
        coefficients=fit.x.reshape(z.shape[1], 4).tolist(),
        prior=((np.bincount(labels["train"], minlength=4) + 1) / (len(y) + 4)).tolist(),
        optimization={
            "success": bool(fit.success),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "gradient_max": gradient,
            "objective": float(fit.fun),
            "alpha": 1,
            "intercept_penalty": 0,
        },
    )
    write(destination / "parameters.json", parameters)
    write(destination / "admission.json", admission)
    if not fit.success or gradient > 1e-3:
        raise RuntimeError("control optimizer did not satisfy the frozen criterion")
    for split, subset in rows.items():
        matrix = np.column_stack([np.ones(len(subset)), (features(subset) - center) / scale])
        for method, p in (
            ("prior", np.broadcast_to(parameters["prior"], (len(subset), 4))),
            ("composition", softmax(matrix @ fit.x.reshape(z.shape[1], 4), axis=1)),
        ):
            with gzip.open(destination / f"{method}-{split}-predictions.jsonl.gz", "wt") as out:
                for i, row in enumerate(subset):
                    record = dict(
                        observation_id=row.observation_id,
                        group_id=row.group_id,
                        split=split,
                        observation=row.to_dict(),
                        species=list(SPECIES),
                        identity_task=next(iter(row.identity_labels)),
                        target=int(labels[split][i]),
                        probabilities=p[i].tolist(),
                        prediction=int(p[i].argmax()),
                        loss=float(-np.log(max(p[i, labels[split][i]], np.finfo(float).tiny))),
                    )
                    out.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("Saved fixed prior and composition controls", flush=True)


def fit(condition):
    predecessor = (
        ROOT
        / "experiments/2026-10-26_1610_codex_ic50-convergence"
        / "reproduce/continuation-state-attempt02.json"
    )
    if read(predecessor)["status"] != "completed":
        raise RuntimeError("IC50 family must finish and verify before neural species fitting")
    output = RAW / condition
    summary = fit_index(index_for(condition), output, device="cpu")
    target = FAMILY / "results" / condition
    target.mkdir(exist_ok=False)
    for name in (
        "summary.json",
        "coverage.json",
        "history.json",
        "admission.json",
        "resolved-config.json",
    ):
        shutil.copyfile(output / name, target / name)
    for path in output.glob("*-predictions.jsonl"):
        with (
            path.open("rb") as source,
            gzip.open(target / (path.name + ".gz"), "wb") as destination,
        ):
            shutil.copyfileobj(source, destination)
    write(
        target / "artifacts.json",
        {
            str(p.relative_to(output)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != ".training.lock"
        },
    )
    print(
        json.dumps(
            {
                "condition": condition,
                "selected_epoch": summary["selected_epoch"],
                "coverage": summary["coverage_gate"]["status"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "baselines", *CONDITIONS))
    args = parser.parse_args()
    torch.set_num_threads(4 if args.action in CONDITIONS else 1)
    torch.set_num_interop_threads(1)
    with guarded(args.action, 10800 if args.action in CONDITIONS else 600):
        if args.action == "prepare":
            prepare()
        elif args.action == "baselines":
            baselines()
        else:
            fit(args.action)


if __name__ == "__main__":
    main()
