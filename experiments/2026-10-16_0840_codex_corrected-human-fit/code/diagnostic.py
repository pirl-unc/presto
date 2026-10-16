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
