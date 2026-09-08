"""Effective MIL observations shared by loss, support counting and holdout export.

Assay descriptors select output columns only. A bag has one response label;
its candidate molecules never inherit individual positive response labels.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..data.vocab import (
    TCELL_ASSAY_METHODS,
    TCELL_ASSAY_READOUTS,
    TCELL_APC_TYPES,
    TCELL_CULTURE_CONTEXTS,
    TCELL_STIM_CONTEXTS,
    TCELL_PEPTIDE_FORMATS,
)


def get_mil_channel(
    batch,
    prefix: str,
) -> Optional[Dict[str, Any]]:
    channel = {
        "pep_tok": getattr(batch, f"{prefix}_pep_tok", None),
        "mhc_a_tok": getattr(batch, f"{prefix}_mhc_a_tok", None),
        "mhc_b_tok": getattr(batch, f"{prefix}_mhc_b_tok", None),
        "mhc_class": getattr(batch, f"{prefix}_mhc_class", None),
        "species": getattr(batch, f"{prefix}_species", None),
        "flank_n_tok": getattr(batch, f"{prefix}_flank_n_tok", None),
        "flank_c_tok": getattr(batch, f"{prefix}_flank_c_tok", None),
        "flank_n_is_terminus": getattr(batch, f"{prefix}_flank_n_is_terminus", None),
        "flank_c_is_terminus": getattr(batch, f"{prefix}_flank_c_is_terminus", None),
        "instance_to_bag": getattr(batch, f"{prefix}_instance_to_bag", None),
        "bag_label": getattr(batch, f"{prefix}_bag_label", None),
        "bag_sample_ids": getattr(batch, f"{prefix}_bag_sample_ids", []),
        "bag_sample_indices": getattr(batch, f"{prefix}_bag_sample_indices", []),
        "tcell_context": getattr(batch, f"{prefix}_context", {}),
        "provenance": getattr(batch, f"{prefix}_provenance", None),
        "machinery_idx": getattr(batch, f"{prefix}_machinery_idx", None),
    }
    required = (
        channel["pep_tok"],
        channel["mhc_a_tok"],
        channel["mhc_b_tok"],
        channel["instance_to_bag"],
        channel["bag_label"],
    )
    if all(value is None for value in required):
        return None
    if any(value is None for value in required):
        raise ValueError(f"Incomplete MIL channel: {prefix}")
    return channel


def slice_mil_channel(
    channel: Dict[str, Any],
    keep: torch.Tensor,
) -> Dict[str, Any]:
    keep_list = keep.tolist()
    sliced = dict(channel)
    for key in (
        "pep_tok",
        "mhc_a_tok",
        "mhc_b_tok",
        "flank_n_tok",
        "flank_c_tok",
        "flank_n_is_terminus",
        "flank_c_is_terminus",
        "instance_to_bag",
        "machinery_idx",
    ):
        value = channel.get(key)
        if isinstance(value, torch.Tensor):
            sliced[key] = value[keep]
    for key in ("mhc_class", "species"):
        value = channel.get(key)
        if isinstance(value, list):
            sliced[key] = [value[i] for i in keep_list]
    # Provenance is per-instance too. `sliced = dict(channel)` shallow-copies
    # it, so without this the capped forward gets full-length condition
    # tensors against truncated inputs and dies in torch.cat -- which any run
    # with max_mil_instances set and a bag larger than the cap would hit.
    provenance = channel.get("provenance")
    if isinstance(provenance, dict):
        sliced["provenance"] = {
            name: (tensor[keep] if isinstance(tensor, torch.Tensor) else tensor)
            for name, tensor in provenance.items()
        }
    context = channel.get("tcell_context")
    if isinstance(context, dict):
        sliced["tcell_context"] = {name: value[keep] for name, value in context.items()}
    return sliced


def run_mil_forward(
    model,
    *,
    channel: Dict[str, Any],
    device: str,
    tcell_context: Optional[Dict[str, torch.Tensor]] = None,
    provenance: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    # Declared per-instance machinery. Omitting it makes the model fall back to
    # a threshold on *predicted* class, which now feeds the elution loss via
    # the excision -> presentation edge.
    channel_machinery = channel.get("machinery_idx")
    return model(
        pep_tok=channel["pep_tok"].to(device),
        mhc_a_tok=channel["mhc_a_tok"].to(device),
        mhc_b_tok=channel["mhc_b_tok"].to(device),
        mhc_class=channel["mhc_class"],
        species=channel["species"],
        flank_n_tok=(
            channel["flank_n_tok"].to(device)
            if isinstance(channel.get("flank_n_tok"), torch.Tensor)
            else None
        ),
        flank_c_tok=(
            channel["flank_c_tok"].to(device)
            if isinstance(channel.get("flank_c_tok"), torch.Tensor)
            else None
        ),
        flank_n_is_terminus=(
            channel["flank_n_is_terminus"].to(device)
            if channel.get("flank_n_is_terminus") is not None
            else None
        ),
        flank_c_is_terminus=(
            channel["flank_c_is_terminus"].to(device)
            if channel.get("flank_c_is_terminus") is not None
            else None
        ),
        # tcell_context is deliberately not forwarded, matching the row path
        # and both holdout forwards. Passing it here would make predict_panel
        # sweep from the observed context on the bag path and from the
        # all-unknown baseline everywhere else, so the same panel outputs
        # would be two different functions averaged into one loss.
        machinery=(
            channel_machinery.to(device) if isinstance(channel_machinery, torch.Tensor) else None
        ),
        provenance=(
            {name: value.to(device) for name, value in provenance.items()} if provenance else None
        ),
    )


@dataclass(frozen=True)
class MILTaskSpec:
    name: str
    output_path: tuple[str, ...]
    axis: str = ""
    columns: tuple[str, ...] = ()
    base_weight: float = 1.0

    @property
    def selector_key(self):
        return f"{self.axis}_idx" if self.axis else ""


MIL_TASKS = {
    "mil": (
        MILTaskSpec("elution", ("elution_logit",)),
        MILTaskSpec("presentation", ("presentation_logit",)),
        # Preserve the existing alias objective and its weight.
        MILTaskSpec("ms", ("ms_logit",)),
    ),
    "tcell_mil": (
        MILTaskSpec("tcell_mil", ("tcell_logit",)),
        MILTaskSpec("immunogenicity_mil", ("immunogenicity_logit",)),
        *(
            MILTaskSpec(f"tcell_{axis}_mil", ("tcell_panel_logits", axis), axis, tuple(columns))
            for axis, columns in (
                ("assay_method", TCELL_ASSAY_METHODS),
                ("assay_readout", TCELL_ASSAY_READOUTS),
                ("apc_type", TCELL_APC_TYPES),
                ("culture_context", TCELL_CULTURE_CONTEXTS),
                ("stim_context", TCELL_STIM_CONTEXTS),
                ("peptide_format", TCELL_PEPTIDE_FORMATS),
            )
        ),
    ),
}
MIL_TASK_BASE_WEIGHTS = {
    spec.name: spec.base_weight for specs in MIL_TASKS.values() for spec in specs
}
# Memory control for final evaluation; it never discards an instance.
DEFAULT_MIL_EVAL_CHUNK_SIZE = 128


@dataclass(frozen=True)
class MILTarget:
    spec: MILTaskSpec
    labels: torch.Tensor
    mask: torch.Tensor
    selectors: Optional[torch.Tensor]
    instance_counts: torch.Tensor

    @property
    def support(self) -> int:
        return int(self.mask.sum().item())


@dataclass(frozen=True)
class MILPrediction:
    target: MILTarget
    probabilities: torch.Tensor
    instance_probability_sums: torch.Tensor
    evaluated_counts: torch.Tensor

    @property
    def logits(self):
        return torch.logit(self.probabilities.clamp(1e-7, 1 - 1e-7))

    @property
    def losses(self):
        # Match mil_bag_loss's existing clipping and probability-space BCE.
        p = self.probabilities.clamp(1e-7, 1 - 1e-7)
        y = self.target.labels
        return -y * torch.log(p) - (1 - y) * torch.log(1 - p)

    def loss(self):
        return self.losses[self.target.mask].mean()


def resolve_mil_targets(channel, specs) -> Dict[str, MILTarget]:
    """Resolve bag labels/selectors/support without evaluating a model.

    Targets are resolved before training sampling. Every selector must be
    constant within its bag, including candidates omitted by a training cap.
    Unknown column zero is explicitly unsupported; named OTHER is supported.
    """
    labels = channel["bag_label"].reshape(-1).float()
    membership = channel["instance_to_bag"].long()
    n_bags = labels.numel()
    if not n_bags:
        return {}
    if (
        membership.ndim != 1
        or membership.numel() != channel["pep_tok"].shape[0]
        or membership.numel() == 0
        or bool((membership < 0).any())
        or bool((membership >= n_bags).any())
    ):
        raise ValueError("Invalid MIL instance-to-bag membership")
    counts = torch.bincount(membership, minlength=n_bags)
    if bool((counts == 0).any()):
        raise ValueError("MIL bags must contain at least one instance")
    targets = {}
    for spec in specs:
        selectors = None
        mask = torch.ones_like(labels, dtype=torch.bool)
        if spec.axis:
            index = channel.get("tcell_context", {}).get(spec.selector_key)
            if index is None:
                index = torch.zeros_like(membership)
            index = index.reshape(-1).long().to(labels.device)
            if (
                index.numel() != membership.numel()
                or bool((index < 0).any())
                or bool((index >= len(spec.columns)).any())
            ):
                raise ValueError(f"Invalid MIL selector {spec.selector_key}")
            selectors = torch.full(
                (n_bags,), len(spec.columns), device=labels.device, dtype=torch.long
            )
            selectors.scatter_reduce_(0, membership, index, reduce="amin", include_self=True)
            maximum = torch.zeros_like(selectors).scatter_reduce_(
                0, membership, index, reduce="amax", include_self=True
            )
            if not torch.equal(selectors, maximum):
                raise ValueError(f"Inconsistent {spec.selector_key} within a MIL bag")
            mask = selectors != 0
        targets[spec.name] = MILTarget(spec, labels, mask, selectors, counts)
    return targets


def predict_mil_channel(
    model, *, channel, targets, device, chunk_size=0
) -> Dict[str, MILPrediction]:
    """Forward complete/capped channel and accumulate bag sufficient statistics.

    Only selected scalar predictions survive each forward. Float32 log-survival
    sums implement the same clamped Noisy-OR as the original dense bag loss,
    without a bag-by-largest-bag matrix. Chunking is intended for model.eval();
    dropout makes training forwards depend on chunk boundaries.
    """
    n_instances = channel["pep_tok"].shape[0]
    if not targets or not n_instances:
        return {}
    chunk_size = int(chunk_size) if chunk_size > 0 else n_instances
    active = {name: target for name, target in targets.items() if target.support > 0}
    if not active:
        return {}
    n_bags = next(iter(targets.values())).labels.numel()
    membership = channel["instance_to_bag"].to(device=device, dtype=torch.long)
    counts = torch.bincount(membership, minlength=n_bags)
    if counts.numel() != n_bags or bool((counts == 0).any()):
        raise ValueError("MIL prediction channel dropped a bag")
    log_survival, probability_sums = {}, {}
    available = None
    for start in range(0, n_instances, chunk_size):
        keep = torch.arange(
            start, min(start + chunk_size, n_instances), device=channel["pep_tok"].device
        )
        chunk = slice_mil_channel(channel, keep)
        outputs = run_mil_forward(
            model, channel=chunk, device=device, provenance=chunk.get("provenance")
        )
        bag_index = membership[keep]
        present = set()
        for name, target in active.items():
            logits = outputs
            for part in target.spec.output_path:
                logits = logits.get(part) if isinstance(logits, dict) else None
            # Some model configurations omit optional heads. The target view
            # still reports their observed support for the coverage registry.
            if logits is None:
                continue
            if not isinstance(logits, torch.Tensor):
                raise ValueError(f"MIL output {target.spec.output_path} is not a tensor")
            present.add(name)
            if target.selectors is not None:
                if logits.ndim != 2 or logits.shape[1] != len(target.spec.columns):
                    raise ValueError(f"Invalid MIL panel shape for {name}: {tuple(logits.shape)}")
                logits = logits.gather(1, target.selectors[bag_index].unsqueeze(1))
            if logits.numel() != keep.numel():
                raise ValueError(f"MIL output {name} must have one selected logit per instance")
            probabilities = logits.reshape(-1).float().sigmoid()
            survival = torch.log1p(-probabilities.clamp(0, 1 - 1e-7))
            logs = probabilities.new_zeros(n_bags).index_add(0, bag_index, survival)
            sums = probabilities.new_zeros(n_bags).index_add(0, bag_index, probabilities)
            log_survival[name] = log_survival.get(name, 0) + logs
            probability_sums[name] = probability_sums.get(name, 0) + sums
        if available is not None and available != present:
            raise ValueError("MIL output availability changed between forward chunks")
        available = present
        del outputs
    return {
        name: MILPrediction(active[name], -torch.expm1(logs), probability_sums[name], counts)
        for name, logs in log_survival.items()
    }
