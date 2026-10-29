"""Freeze the verified data revision and require the same shared training recipe."""

from pathlib import Path
import hashlib
import json
import math
import time

F = Path(__file__).resolve().parents[1]
PARENT = F.parent / "2026-10-28_1540_codex_temporal-multi-output"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = read(F / "results/prepared.json")
    parent = read(PARENT / "reproduce/training-contract.json")
    assert p["status"] == "prepared" and p["verification"]["status"] == "passed"
    assert p["model"] == parent["model"]
    changes = [
        name
        for name, item in p["verification"]["input_receipts"].items()
        if item["before_sha256"] != item["after_sha256"]
    ]
    assert sorted(changes) == [
        "hitlist-v4-iedb-train.jsonl",
        "hitlist_ligand_background-train.jsonl",
    ]
    review = p["source_review"]
    assert review["revision"] == "source-review-v2"
    assert review["removed_rows"] == 618 and review["inferred_bounds"] == {">=": 482, "<=": 24}
    assert review["recovered_positive_rows"] == review["added_background_rows"] == 13549
    assert review["removed_generated_training_rows"] == 35
    assert review["additional_recovered_holdout_embargo"] == 1
    assert review["new_blacklist_sequences"] == 292
    assert review["suspect_generated_test_rows"] == 3
    assert review["blacklist_audit_sha256"] == sha(F / "results/blacklist-audit.json")
    assert review["recovered_positive_blacklist_verified"]
    assert review["recovered_holdout_embargo_verified"]
    assert p["steps"] == math.ceil(p["weighted_epoch_examples"] / 128)
    for key, count in p["train_strata"].items():
        assert p["epoch_quotas"][key] >= count
    for key, count in parent["counts"].items():
        if not key.startswith("train:"):
            assert p["counts"][key] == count
    for name, spec in p["manifests"].items():
        manifest = p["review_metadata"][name + ".json"]
        serialized = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        assert hashlib.sha256(serialized).hexdigest() == spec["sha256"]
        assert manifest["model"] == parent["model"]
        old = read(PARENT / "reproduce/manifests" / ("wide-multi-s" + str(spec["seed"]) + ".json"))
        for key, value in old["training"].items():
            if key not in {"max_steps", "stratum_weights"}:
                assert manifest["training"][key] == value
    current_pins = read(F / "reproduce/runtime-pins.json")
    old_pins = read(PARENT / "reproduce/runtime-pins.json")
    differences = sorted(
        name
        for name, digest in current_pins["files"].items()
        if old_pins["files"].get(name) != digest
    )
    assert differences == [
        "data/measurement_curation.py",
        "data/modular_hitlist.py",
        "data/modular_sources.py",
    ]
    target = F / "reproduce/training-contract.json"
    assert not target.exists()
    contract = dict(
        status="frozen",
        family=F.name,
        frozen_unix=time.time(),
        parent_family=PARENT.name,
        prepared_record_sha256=p["prepared_record_sha256"],
        manifests=p["manifests"],
        model=p["model"],
        counts=p["counts"],
        train_strata=p["train_strata"],
        epoch_quotas=p["epoch_quotas"],
        max_steps=p["steps"],
        weighted_epoch_examples=p["weighted_epoch_examples"],
        background_contract=p["background_contract"],
        verification=p["verification"],
        source_review=review,
        canonical_preflight="pending_in_trainers",
        runtime_pins_sha256=sha(F / "reproduce/runtime-pins.json"),
        runtime_changes_from_parent=differences,
        requested_gpu="H100!",
        max_concurrent_gpus=2,
        seed_selection="minimum equal-endpoint validation loss; seed breaks ties",
        test_used_for_selection=False,
        pretraining=False,
        interpretation=(
            "Repeated fixed previously inspected holdouts; data revision and recomputed "
            "complete-epoch quotas; identical model/optimizer/loss"
        ),
    )
    target.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    tests = read(F / "results/tests/receipt.json")
    assert tests["status"] == "passed"
    for suite in tests["suites"]:
        log = F / suite["path"]
        assert sha(log) == suite["sha256"]
        assert str(suite["passed"]) + " passed" in log.read_text()
    assert sum(s["passed"] for s in tests["suites"]) == 172
    readiness = dict(
        status="passed",
        training_contract_sha256=sha(target),
        partition_verification_sha256=sha(F / "results/partition-verification.json"),
        source_hashes={str(path.relative_to(F)): sha(path) for path in (F / "code").glob("*.py")},
        tests=tests,
        full_preflight="mandatory in each worker before optimizer construction",
    )
    (F / "results/launch-readiness.json").write_text(
        json.dumps(readiness, indent=2, allow_nan=False) + "\n"
    )
    print("Frozen", p["steps"], "updates/seed and", sum(p["train_strata"].values()), "TRAIN rows")


if __name__ == "__main__":
    main()
