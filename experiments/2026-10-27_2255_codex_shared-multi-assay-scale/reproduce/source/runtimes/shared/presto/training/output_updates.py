"""Observe column gradients and optimizer changes without changing optimization."""

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from ..data.vocab import EXCISION_MACHINERY, TCELL_CULTURE_CONTEXTS
from .mil import MIL_TASKS, get_mil_channel, resolve_mil_targets, mil_observation_slots
from .output_contract import (
    OutputConfiguration,
    build_output_contract,
    declared_parameter_rows,
    label_evidence,
)
from .supervision import resolve_row_targets


def _row_vectors(tensor):
    return tensor.detach().reshape(tensor.shape[0], -1).cpu().double()


def _row_sums(tensor):
    rows = _row_vectors(tensor)
    finite = torch.isfinite(rows)
    return torch.where(finite, rows, 0).abs().sum(1).tolist(), (~finite).sum(1).tolist()


def _initial_rows(parameter):
    rows = _row_vectors(parameter)
    return [
        {
            "initial_nonzero_elements": int(nonzero),
            "initial_nonfinite_elements": int(invalid),
            "initial_l2": float(norm) if invalid == 0 else None,
        }
        for nonzero, invalid, norm in zip(
            (rows != 0).sum(1).tolist(),
            (~torch.isfinite(rows)).sum(1).tolist(),
            rows.norm(dim=1).tolist(),
        )
    ]


class OutputUpdateTracker:
    """One tracker for a training run; each train_epoch brackets its batches.

    Output hooks observe derivatives before PCGrad projection. Optimizer hooks
    observe the final gradient supplied to the actual optimizer (after clipping
    or projection) and the resulting parameter value. Neither hook returns a
    replacement value, mutates a tensor, or computes an additional backward.
    """

    def __init__(self, model, optimizer, *, output_dir=None, contract=None):
        self.model = model
        self.raw_model = getattr(model, "_orig_mod", model)
        self.optimizer = optimizer
        self.contract = contract or build_output_contract(
            OutputConfiguration.from_object(self.raw_model)
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.declarations = declared_parameter_rows(self.contract, self.raw_model)
        self._parameters = dict(self.raw_model.named_parameters())
        self._tracked_names = {decl.parameter for decl in self.declarations}
        # These rows intentionally have no dedicated public output column.
        self._extra = {}
        for name, columns, interpretation in (
            ("excision_head.p1_profile_c", EXCISION_MACHINERY, "machinery_profile"),
            (
                "tcell_assay_head.duration_default.weight",
                TCELL_CULTURE_CONTEXTS,
                "shared_duration_default",
            ),
        ):
            self._tracked_names.add(name)
            self._extra[name] = (tuple(columns), interpretation)
        self.parameter_rows = {}
        for name in sorted(self._tracked_names):
            parameter = self._parameters[name]
            self.parameter_rows[name] = [
                {
                    **initial,
                    "initial_requires_grad": parameter.requires_grad,
                    "steps_in_optimizer": 0,
                    "steps_frozen": 0,
                    "steps_with_gradient": 0,
                    "steps_updated": 0,
                    "steps_updated_without_gradient": 0,
                    "gradient_abs_sum": 0.0,
                    "update_abs_sum": 0.0,
                    "nonfinite_gradient_elements": 0,
                    "nonfinite_update_elements": 0,
                }
                for initial in _initial_rows(parameter)
            ]
        self.cells = {}
        for endpoint, spec in self.contract.outputs.items():
            for column in spec.columns or ("",):
                self._cell(endpoint, column)
        self.epoch = 0
        self.epochs_observed = set()
        self.batches_started = self.batches_completed = self.batches_aborted = (
            self.optimizer_steps
        ) = 0
        self._active = False
        self._closed = False
        self._tensor_hooks = []
        self._hooks = [
            model.register_forward_hook(self._forward_hook),
            optimizer.register_step_pre_hook(self._optimizer_before),
            optimizer.register_step_post_hook(self._optimizer_after),
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _cell(self, endpoint, column):
        return self.cells.setdefault(
            (endpoint, column),
            {
                "endpoint": endpoint,
                "column": column,
                "forward_calls": 0,
                "differentiable_forward_calls": 0,
                "batches_with_output_gradient": 0,
                "output_gradient_abs_sum": 0.0,
                "nonfinite_output_gradient_elements": 0,
                "label_observations_seen": 0,
                "positive_labels_seen": 0,
                "batches_with_labels": 0,
                "batches_with_direct_labels": 0,
                "source_evidence": Counter(),
            },
        )

    @contextmanager
    def observe_batch(self, batch):
        if self._active or self._closed:
            raise RuntimeError("Output tracker is already observing a batch or closed")
        self._active = True
        self.batches_started += 1
        self.epochs_observed.add(self.epoch)
        self._gradient_cells = set()
        self._labels = defaultdict(Counter)
        self._label_positive = Counter()
        self._snapshots = {}
        aborted = True
        try:
            self._observe_labels(batch)
            yield
            aborted = False
        finally:
            for handle in self._tensor_hooks:
                handle.remove()
            self._tensor_hooks.clear()
            self._active = False
            self.batches_aborted += int(aborted)
            self.batches_completed += int(not aborted)
            for key in self._gradient_cells:
                self._cell(*key)["batches_with_output_gradient"] += 1
            for key, evidence in self._labels.items():
                cell = self._cell(*key)
                cell["label_observations_seen"] += sum(evidence.values())
                cell["positive_labels_seen"] += self._label_positive[key]
                cell["batches_with_labels"] += 1
                cell["batches_with_direct_labels"] += int(
                    any(role == "direct" for role, _, _ in evidence)
                )
                cell["source_evidence"].update(evidence)
            self._snapshots.clear()

    def _observe_labels(self, batch):
        n_rows = batch.pep_tok.shape[0]
        metadata = getattr(batch, "sample_evidence", [])
        if metadata and len(metadata) != n_rows:
            raise ValueError("Output-update evidence metadata does not match batch rows")
        samples = (
            [SimpleNamespace(**value) for value in metadata]
            if metadata
            else [
                SimpleNamespace(
                    sample_source="unknown",
                    synthetic_kind=None,
                    target_provenance={},
                    binding_assay_type=None,
                    bind_measurement_type=None,
                )
                for _ in range(n_rows)
            ]
        )
        seen = set()

        def add(identity, row, column, target, positive, occurrence=0):
            objective = self.contract.objectives[identity]
            key = objective.endpoint, column
            # Intentional alias/duplicate losses are one label exposure.
            observation = (
                *key,
                row,
                objective.source_target,
                float(target),
                objective.channel,
                occurrence,
            )
            if observation in seen:
                return
            seen.add(observation)
            evidence = label_evidence(objective, samples[row])
            self._labels[key][(evidence.role, evidence.family, evidence.raw_source)] += 1
            self._label_positive[key] += int(positive)

        for name, target in resolve_row_targets(batch).items():
            spec = target.spec
            for index in (target.mask > 0).nonzero().flatten().tolist():
                row, value = int(target.source_rows[index]), float(target.target[index])
                if spec.class_names:
                    for class_index, column in enumerate(spec.class_names):
                        add(f"row:{name}", row, column, value, value == class_index)
                    continue
                column = ""
                if target.selectors is not None:
                    column = spec.columns[int(target.selectors[index])]
                elif target.components is not None:
                    column = spec.component_names[int(target.components[index])]
                add(f"row:{name}", row, column, value, spec.loss_type == "bce" and value > 0.5)
                if spec.loss_type == "ce" and not spec.class_names:
                    add(f"row:{name}", row, f"position:{int(value)}", value, True)
        for prefix, specs in MIL_TASKS.items():
            channel = get_mil_channel(batch, prefix)
            if channel is None:
                continue
            slots = mil_observation_slots(channel, source_rows=n_rows)
            for name, target in resolve_mil_targets(channel, specs).items():
                for index in target.mask.nonzero().flatten().tolist():
                    row = int(channel["bag_sample_indices"][index])
                    column = (
                        target.spec.columns[int(target.selectors[index])]
                        if target.selectors is not None
                        else ""
                    )
                    value = float(target.labels[index])
                    add(
                        f"{prefix}:{name}", row, column, value, value > 0.5, occurrence=slots[index]
                    )

    def _forward_hook(self, _model, _inputs, outputs):
        if not self._active:
            return
        for endpoint, spec in self.contract.outputs.items():
            tensor = outputs
            for part in endpoint.split("."):
                tensor = tensor.get(part) if isinstance(tensor, dict) else None
            if not isinstance(tensor, torch.Tensor):
                continue
            columns = spec.columns or ("",)
            by_column = bool(spec.columns)
            if spec.shape == "positions" and tensor.ndim == 2:
                columns = tuple(f"position:{index}" for index in range(tensor.shape[1]))
                by_column = True
            if by_column and (tensor.ndim != 2 or tensor.shape[1] != len(columns)):
                raise ValueError(f"Output tracker observed a changed column shape for {endpoint}")
            for column in columns:
                cell = self._cell(endpoint, column)
                cell["forward_calls"] += 1
                cell["differentiable_forward_calls"] += int(tensor.requires_grad)
            if not tensor.requires_grad:
                continue

            def observe(gradient, endpoint=endpoint, columns=columns, by_column=by_column):
                values = gradient.T if by_column else gradient.reshape(1, -1)
                totals, invalids = _row_sums(values)
                for column, total, invalid in zip(columns, totals, invalids):
                    cell = self._cell(endpoint, column)
                    cell["output_gradient_abs_sum"] += total
                    cell["nonfinite_output_gradient_elements"] += invalid
                    if total > 0:
                        self._gradient_cells.add((endpoint, column))
                # Returning None leaves the derivative unchanged.

            self._tensor_hooks.append(tensor.register_hook(observe))

    def _optimizer_before(self, optimizer, _args, _kwargs):
        if not self._active:
            return
        in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self._snapshots = {}
        for name in self._tracked_names:
            parameter = self._parameters[name]
            self._snapshots[name] = parameter.detach().clone()
            gradient, invalids = (
                _row_sums(parameter.grad)
                if parameter.grad is not None
                else ([0.0] * parameter.shape[0], [0] * parameter.shape[0])
            )
            for row, total, invalid in zip(self.parameter_rows[name], gradient, invalids):
                row["steps_in_optimizer"] += int(id(parameter) in in_optimizer)
                row["steps_frozen"] += int(not parameter.requires_grad)
                row["steps_with_gradient"] += int(total > 0)
                row["gradient_abs_sum"] += total
                row["nonfinite_gradient_elements"] += invalid
            self._snapshots[(name, "gradient")] = gradient

    def _optimizer_after(self, _optimizer, _args, _kwargs):
        if not self._active:
            return
        self.optimizer_steps += 1
        for name in self._tracked_names:
            parameter = self._parameters[name]
            delta, invalids = _row_sums(parameter.detach() - self._snapshots[name])
            for row, change, gradient, invalid in zip(
                self.parameter_rows[name], delta, self._snapshots[(name, "gradient")], invalids
            ):
                row["steps_updated"] += int(change > 0)
                row["steps_updated_without_gradient"] += int(change > 0 and gradient == 0)
                row["update_abs_sum"] += change
                row["nonfinite_update_elements"] += invalid
        self._snapshots.clear()

    def report(self):
        mappings = defaultdict(list)
        for decl in self.declarations:
            for index, column in enumerate(decl.columns):
                mappings[(decl.endpoint, column)].append(
                    {
                        "parameter": decl.parameter,
                        "row": index,
                        "interpretation": decl.interpretation,
                        **self.parameter_rows[decl.parameter][index],
                    }
                )
        cells = []
        for key, value in sorted(self.cells.items()):
            cells.append(
                {
                    **value,
                    "source_evidence": [
                        {
                            "role": role,
                            "family": family,
                            "source": source,
                            "observations_seen": count,
                        }
                        for (role, family, source), count in sorted(
                            value["source_evidence"].items()
                        )
                    ],
                    "parameter_mapping": "column_parameters"
                    if mappings[key]
                    else "no_dedicated_row_declared",
                    "parameter_rows": mappings[key],
                }
            )
        extra = []
        for name, (columns, interpretation) in self._extra.items():
            pinned = (
                self.raw_model.excision_head.pinned_mask.detach().cpu().tolist()
                if interpretation == "machinery_profile"
                else [False] * len(columns)
            )
            for index, column in enumerate(columns):
                extra.append(
                    {
                        "parameter": name,
                        "row": index,
                        "column": column,
                        "interpretation": interpretation,
                        "replaced_by_fixed_rule": pinned[index],
                        **self.parameter_rows[name][index],
                    }
                )
        return {
            "schema_version": 1,
            "configuration": asdict(self.contract.configuration),
            "batches_started": self.batches_started,
            "batches_completed": self.batches_completed,
            "batches_aborted": self.batches_aborted,
            "optimizer_steps": self.optimizer_steps,
            "epochs_observed": sorted(self.epochs_observed),
            "outputs": cells,
            "additional_parameter_rows": extra,
            "interpretation": [
                "Label counts record training exposures, not distinct observations or accuracy.",
                "Output derivatives precede PCGrad; parameter gradients reach the optimizer.",
                "Updates and shared gradients do not establish direct supervision or causality.",
                "Zero-gradient rows can change through weight decay or optimizer state.",
                "Core position labels count gold-start incidence, not independent binary labels.",
            ],
        }

    def write(self):
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "output_updates.json"
        path.write_text(json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

    def close(self):
        if self._closed:
            return
        for handle in self._hooks + self._tensor_hooks:
            handle.remove()
        self._hooks.clear()
        self._tensor_hooks.clear()
        self._closed = True
        self.write()
