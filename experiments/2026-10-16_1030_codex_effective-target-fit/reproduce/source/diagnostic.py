"""Observe the same training-only IC50 cohort before and after optimization."""

import functools
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from baselines import ENDPOINT
from presto.data.collate import PrestoCollator
from presto.scripts import train_synthetic
from presto.training.checkpointing import (
    load_model_from_checkpoint,
    save_model_checkpoint,
)
from presto.training.holdout_eval import (
    collect_holdout_predictions,
    write_holdout_artifacts,
)
from torch.utils.data import DataLoader


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate_training(model, samples, work, split):
    loader = DataLoader(
        samples,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=PrestoCollator(),
        generator=torch.Generator().manual_seed(1042),
    )
    specs = [s for s in train_synthetic.LOSS_TASK_SPECS if s.name == "binding_ic50"]
    if len(specs) != 1:
        raise ValueError("Direct IC50 loss objective changed")
    was_training = model.training
    try:
        with preserve_rng():
            predictions = collect_holdout_predictions(
                model,
                loader,
                "cpu",
                specs,
                lambda model, batch: model(**batch.model_inputs()),
                train_synthetic._resolve_task_prediction,
                train_synthetic._get_batch_target,
                train_synthetic._get_batch_mask,
                train_synthetic._get_batch_qual,
            )
            write_holdout_artifacts(
                work,
                predictions,
                split=split,
                extra_summary={
                    "scope": "Training-only diagnostic; never held-out selection evidence."
                },
            )
    finally:
        model.train(was_training)


def install(train_iedb, work, expected_train_ids, config):
    original = train_iedb.train_epoch
    captured = {}

    @functools.wraps(original)
    def observe(model, train_loader, *args, **kwargs):
        if "samples" not in captured:
            samples = [s for s in train_loader.dataset if s.source_endpoint == ENDPOINT]
            samples.sort(key=lambda sample: sample.sample_id)
            if len(samples) != 2074 or {s.sample_id for s in samples} != set(expected_train_ids):
                raise ValueError(
                    "Initialization diagnostic differs from frozen training IC50 cohort"
                )
            if any(s.source_partition != "train" for s in samples):
                raise ValueError("Held-out observation entered training diagnostic")
            captured["samples"] = samples
            (Path(work) / "training-diagnostic-membership.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": s.sample_id,
                            "evidence_row_id": s.evidence_row_id,
                            "peptide": s.peptide,
                            "source_partition": s.source_partition,
                        }
                        for s in samples
                    ],
                    indent=2,
                )
                + "\n"
            )
            save_model_checkpoint(
                work / "initial_checkpoint.pt",
                model=model,
                epoch=0,
                step=0,
                train_config=config,
                extra={"state": "Initialized; no optimizer step."},
            )
            evaluate_training(model, samples, work, "train_initial")
        return original(model, train_loader, *args, **kwargs)

    train_iedb.train_epoch = observe

    def finish():
        if "samples" not in captured:
            raise ValueError("No optimization epoch reached the training diagnostic")
        model, payload = load_model_from_checkpoint(work / "checkpoint.pt", map_location="cpu")
        evaluate_training(model, captured["samples"], work, "train_selected")
        return {
            "selected_epoch": payload["epoch"],
            "checkpoint_metrics": payload["metrics"],
            "n_training_ic50_observations": len(captured["samples"]),
            "checkpoint_reloaded_in_fresh_model": True,
        }

    return finish


def install_model_guard(trainer, work, cohort, declaration):
    """Require the complete canonical audit before constructing the first model."""
    import pandas as pd
    from baselines import require_effective_target_contract, write

    original = trainer.Presto

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        require_effective_target_contract(cohort, declaration)
        supported = json.loads((work / "supported_outputs.json").read_text())
        if (
            not supported["requirements_met"]
            or supported["source_contract"] != declaration["source_contract"]
        ):
            raise ValueError(
                "Canonical prospective support gate did not pass before model construction"
            )
        audit = json.loads((work / "affinity_target_audit.json").read_text())
        if audit["policies"] != declaration["source_contract"]["affinity_target_policies"]:
            raise ValueError("Canonical target audit policy differs from the frozen declaration")
        changed = pd.read_csv(
            work / "affinity_target_changes.tsv", sep="\t", keep_default_na=False, dtype=str
        )
        changed = changed[changed.objective_id == "row:binding_ic50"]
        expected = cohort[cohort.physical_value_clipped]
        if not changed.sample_id.is_unique or set(changed.sample_id) != set(expected.sample_id):
            raise ValueError("Canonical clipped IC50 identities differ from the frozen raw labels")
        changed = changed.set_index("sample_id").loc[expected.sample_id]
        for field in ("evidence_row_id", "source_value", "source_units", "source_inequality"):
            if changed[field].tolist() != expected[field].tolist():
                raise ValueError("Canonical clipped IC50 source metadata differs: " + field)
        if not np.allclose(
            changed.effective_target_log10_nM.to_numpy(float),
            expected.target_log10_nM,
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError("Canonical clipped IC50 effective target differs")
        for split in ("train", "val", "test"):
            observed = next(
                r
                for r in audit["counts"]
                if r["split"] == split
                and r["objective_id"] == "row:binding_ic50"
                and not r["column"]
            )
            all_rows = cohort[cohort.source_partition == split]
            clipped = expected[expected.source_partition == split]
            if (
                observed["n_observations"],
                observed["n_clipped_observations"],
                observed["n_clipped_exact_observations"],
            ) != (len(all_rows), len(clipped), int((clipped.qualifier == 0).sum())):
                raise ValueError("Canonical clipped IC50 split counts differ")
        write(
            work / "model-construction-policy-check.json",
            {
                "status": "passed before model construction",
                "n_ic50_observations": len(cohort),
                "n_clipped_observations": len(changed),
                "n_clipped_exact_observations": int((expected.qualifier == 0).sum()),
                "target_policy": declaration["source_contract"]["affinity_target_policies"],
            },
        )
        return original(*args, **kwargs)

    trainer.Presto = guarded
