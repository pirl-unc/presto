"""Independent closure checks for selected checkpoints and exported validation rows."""

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

E = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    result = []
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            observation = row["observation"]
            provenance = observation["provenance"]
            # Full register/APC diagnostics can exceed a gigabyte per export.
            # Retain only the original fields needed for independent checks.
            result.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "observation_id",
                            "endpoint",
                            "mhc_classes",
                            "target",
                            "qualifier",
                            "group_id",
                            "loss",
                        )
                    },
                    "observation": {
                        "split": observation["split"],
                        "peptide": observation["peptide"],
                        "provenance": {
                            "study_id": provenance.get("study_id"),
                            "pmid": provenance.get("pmid"),
                            "raw_record": {"pmid": provenance.get("raw_record", {}).get("pmid")},
                        },
                    },
                    "prediction": row["prediction"],
                }
            )
    return result


def main():
    checks = []
    reference = None
    for folder in sorted((E / "results/fits").glob("*")):
        if folder.name.endswith(("-n16", "-n12", "-n8")) or not (folder / "receipt.json").exists():
            continue
        receipt = json.loads((folder / "receipt.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "raw/summary.json").read_text())
        updates = json.loads((folder / "update-check.json").read_text())
        predictions = records(folder / "raw/validation-predictions.jsonl.gz")
        assert sha(folder / "raw/selected.pt") == receipt["checkpoint_sha256"]
        assert not updates["frozen_parameters"]
        assert summary["coverage_gate"]["status"] == "passed"
        assert manifest["training"]["selection_reduction"] == "class_study_mean"
        assert manifest["training"]["selection_endpoint"] == "elution_observation"
        assert manifest["pretrained"]["freeze"] == []
        assert manifest["training"]["test_policy"] == "no_test_training_diagnostic"
        assert all(r["observation"]["split"] == "validation" for r in predictions)
        keys = {
            r["observation_id"]: (
                r["endpoint"],
                tuple(r["mhc_classes"]),
                r["target"],
                r["qualifier"],
                r["observation"]["peptide"],
                r["group_id"],
            )
            for r in predictions
        }
        assert len(keys) == len(predictions), "duplicate validation examples"
        if reference is None:
            reference = keys
        else:
            assert keys == reference, "different validation populations"
        grouped = defaultdict(lambda: defaultdict(list))
        for r in predictions:
            if r["endpoint"] != "elution_observation":
                continue
            p = r["observation"]["provenance"]
            study = p.get("study_id") or p.get("pmid") or p.get("raw_record", {}).get("pmid")
            assert study is not None
            assert len(set(r["mhc_classes"])) == 1
            grouped[r["mhc_classes"][0]][str(study)].append(r["loss"])
        assert set(grouped) == {"I", "II"}
        per_class = {
            c: sum(sum(v) / len(v) for v in studies.values()) / len(studies)
            for c, studies in grouped.items()
        }
        selected_loss = sum(per_class.values()) / 2
        checkpoint = torch.load(folder / "raw/selected.pt", map_location="cpu", weights_only=True)
        expected_loss = checkpoint["metadata"]["selection_metric"]
        assert abs(selected_loss - expected_loss) < 1e-7
        exposure = json.loads((folder / "actual-training-exposure.json").read_text())
        assert exposure["selected_steps"] == checkpoint["metadata"]["steps"]
        assert (
            exposure["executed"]["draws"] == summary["steps"] * manifest["training"]["batch_size"]
        )
        checks.append(
            {
                "run": folder.name,
                "validation_rows": len(keys),
                "overall_validation_loss": sum(r["loss"] for r in predictions) / len(predictions),
                "class_study_losses": per_class,
                "recomputed_selection_loss": selected_loss,
                "recorded_selection_loss": expected_loss,
                "summary_selection_fields": {
                    k: v for k, v in summary.items() if "selection" in k or "best" in k
                },
                "checkpoint_hash_verified": True,
                "same_examples": True,
                "unfrozen": True,
                "test_accessed": False,
            }
        )
    assert checks, "no completed predictive fits to verify"
    (E / "results/independent-verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    terminal_checks = []
    for folder in sorted((E / "results/evaluation").glob("*-terminal")):
        if not (folder / "receipt.json").exists():
            continue
        from sklearn.metrics import roc_auc_score

        receipt = json.loads((folder / "receipt.json").read_text())
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert sha(folder / "checkpoint.pt") == receipt["checkpoint_sha256"]
        assert checkpoint["metadata"]["steps"] == 6000
        assert receipt["checkpoint_kind"] == "fixed_terminal_6000_updates"
        assert not receipt["test_accessed"] and receipt["original_loss_selection_preserved"]
        assert not checkpoint["frozen_parameters"]
        selected_folder = E / "results/fits" / receipt["parent_run"]
        selected_receipt = json.loads((selected_folder / "receipt.json").read_text())
        assert sha(selected_folder / "raw/selected.pt") == selected_receipt["checkpoint_sha256"]
        selected = torch.load(
            selected_folder / "raw/selected.pt", map_location="cpu", weights_only=True
        )
        assert set(checkpoint["trained_parameters"]) >= set(selected["trained_parameters"])
        for name, features in selected["trained_features"].items():
            assert set(checkpoint["trained_features"][name]) >= set(features)
        for name, mask in selected["element_coverage"]["masks"].items():
            assert torch.all(checkpoint["element_coverage"]["masks"][name] >= mask)
        predictions = records(folder / "validation-predictions.jsonl.gz")
        keys = {
            r["observation_id"]: (
                r["endpoint"],
                tuple(r["mhc_classes"]),
                r["target"],
                r["qualifier"],
                r["observation"]["peptide"],
                r["group_id"],
            )
            for r in predictions
        }
        assert len(keys) == len(predictions) and keys == reference
        assert all(r["observation"]["split"] == "validation" for r in predictions)
        history = json.loads((selected_folder / "raw/history.json").read_text())[-1]
        per_class, aucs = {}, {}
        for cls in ("I", "II"):
            rows = [
                r
                for r in predictions
                if r["endpoint"] == "elution_observation" and set(r["mhc_classes"]) == {cls}
            ]
            aucs[cls] = roc_auc_score([r["target"] for r in rows], [r["prediction"] for r in rows])
            expected = history["validation"]["elution_observation"]["by_class"][cls]
            assert len(rows) == expected["n"] and abs(aucs[cls] - expected["auroc"]) < 1e-6
            studies = defaultdict(list)
            for row in rows:
                p = row["observation"]["provenance"]
                study = p.get("study_id") or p.get("pmid") or p.get("raw_record", {}).get("pmid")
                assert study is not None
                studies[str(study)].append(row["loss"])
            per_class[cls] = sum(sum(v) / len(v) for v in studies.values()) / len(studies)
        terminal_checks.append(
            {
                "run": folder.name,
                "validation_rows": len(keys),
                "steps": 6000,
                "final_history_auroc": aucs,
                "class_study_bce": per_class,
                "equal_class_study_bce": sum(per_class.values()) / 2,
                "overall_validation_loss": sum(r["loss"] for r in predictions) / len(predictions),
                "selected_checkpoint_preserved": True,
                "coverage_superset_verified": True,
                "checkpoint_hash_verified": True,
                "test_accessed": False,
            }
        )
    (E / "results/terminal-independent-verification.json").write_text(
        json.dumps(terminal_checks, indent=2) + "\n"
    )
    print("terminal checkpoints verified:", len(terminal_checks))


if __name__ == "__main__":
    main()
