"""Deterministic code-contract diagnostics; no fitting or quality evaluation.

Run from the repository root with its installed Python environment:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tasks/audits/2026-09-07_output-quality/diagnose.py
"""

import csv
import dataclasses
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.loaders import PrestoDataset
from presto.models.presto import Presto
from presto.scripts.train_iedb import load_records_from_merged_tsv
from presto.scripts.train_synthetic import (
    LOSS_TASK_SPECS,
    _get_batch_mask,
    _get_batch_qual,
    _get_batch_target,
    _resolve_task_prediction,
    compute_loss,
)
from presto.training.data_support import audit_split_support, validate_split_support
from presto.training.holdout_eval import auprc, collect_holdout_predictions

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
A = "ACDEFGHIKLMNPQRS"
B = "TVWYACDEFGHIKLMN"
COLLATOR = PrestoCollator(max_pep_len=12, max_mhc_len=24)


def sample(sample_id, **fields):
    return PrestoSample(
        peptide="SIINFEKL",
        mhc_a=A,
        mhc_b=B,
        mhc_class="I",
        sample_id=sample_id,
        sample_source="audit_fixture",
        **fields,
    )


def collect(model, batch, specs=LOSS_TASK_SPECS):
    return collect_holdout_predictions(
        model,
        [batch],
        "cpu",
        specs,
        lambda model_ref, batch_ref: model_ref(**batch_ref.model_inputs()),
        _resolve_task_prediction,
        _get_batch_target,
        _get_batch_mask,
        _get_batch_qual,
    )


def main():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    evidence = {
        "kind": "deterministic code-contract diagnostics, NOT predictive evaluation",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "model_config": {"d_model": 32, "n_layers": 2, "n_heads": 4, "latent_topology": "expanded"},
        "optimizer_steps": 0,
    }
    specs = []
    for spec in LOSS_TASK_SPECS:
        entry = dataclasses.asdict(spec)
        entry["target_transform"] = bool(spec.target_transform)
        specs.append(entry)
    (OUT / "loss_specs.json").write_text(json.dumps(specs, indent=2) + "\n")
    model = Presto(**evidence["model_config"]).eval()
    batch = COLLATOR([sample("inventory")])
    with torch.no_grad():
        output = model(**batch.model_inputs())
    inventory = []
    seen = {}

    def visit(path, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}.{key}" if path else key, child)
        elif isinstance(value, torch.Tensor):
            inventory.append(
                {
                    "path": path,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "same_tensor_as": seen.get(id(value), ""),
                }
            )
            seen.setdefault(id(value), path)

    visit("", output)
    (OUT / "output_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    evidence["inventory"] = {
        "loss_specs": len(specs),
        "tensor_paths": len(inventory),
        "distinct_tensor_objects": len(seen),
        "binding_panel_columns": sum(
            output[k].shape[1] for k in output if k.startswith("binding_assay_panel_")
        ),
        "tcell_panel_columns": sum(v.shape[1] for v in output["tcell_panel_logits"].values()),
    }

    # Balanced at task level, one-class at each observed method column.
    splits = {
        split: [
            sample(f"{split}-pos", tcell_label=1.0, tcell_assay_method="ELISPOT"),
            sample(f"{split}-neg", tcell_label=0.0, tcell_assay_method="ICS"),
        ]
        for split in ("train", "val", "test")
    }
    support = audit_split_support(splits, collator=COLLATOR)
    validate_split_support(support, require_all_active=True, require_all_active_binary_balance=True)
    evidence["support_gate"] = {
        "passes_all_active_and_binary_balance": True,
        "observed_method_columns": {
            "ELISPOT": {"positive": 1, "negative": 0},
            "ICS": {"positive": 0, "negative": 1},
        },
        "reported_train_targets": support["splits"]["train"]["targets"],
        "binary_targets": support["binary_targets"],
        "mhc_class_has_loss_target_but_absent_from_support": _get_batch_target(
            COLLATOR(splits["train"]), next(s for s in LOSS_TASK_SPECS if s.name == "mhc_class")
        )
        is not None
        and "mhc_class" not in support["splits"]["train"]["targets"],
    }

    # MIL still trains the response scalar, but never the observed assay column.
    mil_sample = sample(
        "tcell-bag",
        tcell_label=1.0,
        tcell_assay_method="ELISPOT",
        use_tcell_pathway_mil=True,
        tcell_mil_mhc_a_list=[A, B],
        tcell_mil_mhc_b_list=[B, A],
        tcell_mil_mhc_class_list=["I", "II"],
        tcell_mil_species_list=["human", "human"],
    )
    mil_batch = COLLATOR([mil_sample])
    model.zero_grad(set_to_none=True)
    loss, terms, _ = compute_loss(model, mil_batch, "cpu")
    loss.backward()
    index = mil_batch.tcell_context["assay_method_idx"].item()
    grad = model.tcell_assay_head.assay_method_embed.weight.grad
    dumped = collect(model, mil_batch)
    mil_support = audit_split_support({"train": [mil_sample]}, collator=COLLATOR)
    evidence["tcell_mil"] = {
        "loss_keys": sorted(terms),
        "selected_method_index": index,
        "selected_method_gradient_abs_sum": float(grad[index].abs().sum()),
        "unknown_method_gradient_abs_sum": float(grad[0].abs().sum()),
        "panel_mask": mil_batch.target_masks["tcell_assay_method"].tolist(),
        "dumped_tcell_rows": len(dumped["tcell"]),
        "support_targets": mil_support["splits"]["train"]["targets"],
    }

    # Known instance probabilities make the row-versus-bag mismatch exact.
    class ConstantElution(torch.nn.Module):
        def forward(self, **kwargs):
            count = kwargs["pep_tok"].shape[0]
            return {
                key: torch.zeros(count, 1)
                for key in ("elution_logit", "presentation_logit", "ms_logit")
            }

    elution = COLLATOR(
        [
            sample(
                "elution-bag",
                elution_label=1.0,
                mil_mhc_a_list=[A, B],
                mil_mhc_b_list=[B, A],
                mil_mhc_class_list=["I", "I"],
            )
        ]
    )
    _, terms, _ = compute_loss(ConstantElution(), elution, "cpu")
    row = collect(ConstantElution(), elution)["elution"].rows()[0]
    evidence["elution_mil"] = {
        "instances": 2,
        "instance_probability": 0.5,
        "training_bag_probability": 0.75,
        "training_bag_loss": float(terms["elution"]),
        "dumped_row_probability": row["y_prob"],
    }

    # Vector BCE flattens the component dimension without expanding identities.
    evidence_sample = sample(
        "receptor-evidence", tcr_evidence_label=1.0, tcr_evidence_method_bins=("multimer_binding",)
    )
    vector_batch = COLLATOR([evidence_sample])
    dumped = collect(model, vector_batch)
    evidence["vector_holdout"] = {
        "mhc_class_target_count": 1,
        "mhc_class_dump_count": len(dumped["mhc_class"]),
        "method_rows": [
            {key: row[key] for key in ("task", "sample_id", "source", "y_true")}
            for row in dumped["tcr_evidence_method"].rows()
        ],
    }

    # Reproduce loss of supplied assay descriptors at the real TSV adapter.
    tsv_row = {
        "peptide": "SIINFEKL",
        "mhc_allele": "HLA-A*02:01",
        "mhc_class": "I",
        "source": "audit_fixture",
        "record_type": "binding",
        "value": "100",
        "value_type": "IC50",
        "qualifier": "0",
        "assay_type": "IC50",
        "assay_method": "purified MHC/direct/fluorescence",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp) / "metadata.tsv"
        with tsv.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tsv_row), delimiter="\t")
            writer.writeheader()
            writer.writerow(tsv_row)
        records = load_records_from_merged_tsv(
            tsv,
            **{
                f"max_{key}": 10
                for key in (
                    "binding",
                    "kinetics",
                    "stability",
                    "processing",
                    "elution",
                    "tcell",
                    "vdjdb",
                )
            },
        )[0]
    assert len(records) == 1
    exact = {
        "HLA-A*02:01": {
            "allele": "HLA-A*02:01",
            "sequence": A * 4 + B * 4,
            "groove1": A * 4,
            "groove2": B * 4,
            "mhc_class": "I",
            "chain": "alpha",
            "groove_status": "complete",
            "source": "audit_fixture",
        }
    }
    data = PrestoDataset(binding_records=records, mhc_exact_inputs=exact)
    metadata_batch = COLLATOR([data[0]])
    evidence["merged_binding_metadata"] = {
        "source_method": tsv_row["assay_method"],
        "record_method": records[0].assay_method,
        "record_assay_type": records[0].assay_type,
        "batch_selectors": {
            key: value.tolist() for key, value in metadata_batch.binding_context.items()
        },
    }

    # One actual source example establishes that this is not only a possible
    # TSV input. This is a bounded read, not a census of current corpus support.
    merged = ROOT / "data/merged_deduped.tsv"
    if merged.exists():
        with merged.open() as handle:
            for line, row in enumerate(csv.DictReader(handle, delimiter="\t"), 2):
                if (
                    row.get("record_type") == "binding"
                    and row.get("value")
                    and row.get("assay_method")
                ):
                    evidence["real_source_example"] = {
                        "path": "data/merged_deduped.tsv",
                        "line": line,
                        "row_sha256": hashlib.sha256(
                            json.dumps(row, sort_keys=True).encode()
                        ).hexdigest(),
                        "fields": {
                            key: row.get(key)
                            for key in (
                                "peptide",
                                "mhc_allele",
                                "value",
                                "value_type",
                                "qualifier",
                                "assay_type",
                                "assay_method",
                                "record_type",
                                "source",
                            )
                        },
                    }
                    break
                if line >= 5000:
                    break

    # A constant score must have AP equal to prevalence, independently of order.
    evidence["average_precision_ties"] = {
        "scores": [0.5, 0.5],
        "expected_average_precision": 0.5,
        "positive_first": auprc(np.array([1.0, 0.0]), np.array([0.5, 0.5])),
        "negative_first": auprc(np.array([0.0, 1.0]), np.array([0.5, 0.5])),
    }

    assert evidence["tcell_mil"]["selected_method_gradient_abs_sum"] == 0
    assert evidence["tcell_mil"]["unknown_method_gradient_abs_sum"] > 0
    assert "tcell_mil" in evidence["tcell_mil"]["loss_keys"]
    assert evidence["tcell_mil"]["dumped_tcell_rows"] == 0
    assert abs(evidence["elution_mil"]["training_bag_loss"] + np.log(0.75)) < 1e-6
    assert evidence["elution_mil"]["dumped_row_probability"] == 0.5
    assert evidence["vector_holdout"]["mhc_class_dump_count"] == 0
    assert [row["sample_id"] for row in evidence["vector_holdout"]["method_rows"]] == [
        "receptor-evidence",
        "",
        "",
    ]
    assert evidence["merged_binding_metadata"]["record_method"] is None
    assert evidence["average_precision_ties"]["positive_first"] == 1.0

    # Preserve provenance for the existing real-data evidence consulted.
    history = ROOT / "experiments/2026-09-05_0957_codex_pr45-integrity-e2e/results"
    historical = {}
    for relative in (
        "experiment_summary.json",
        "merged_preflight/split_support.json",
        "merged_preflight/data_funnel.json",
        "run/split_support.json",
    ):
        path = history / relative
        historical[relative] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contents": json.loads(path.read_text()),
        }
    (OUT / "historical_evidence.json").write_text(json.dumps(historical, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in evidence.items() if key != "dirty_status"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
