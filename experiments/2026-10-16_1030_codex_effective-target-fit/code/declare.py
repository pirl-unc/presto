"""Freeze a new explicit-policy declaration from immutable historical inputs."""

import copy
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import presto
from datacache import install_bundle, materialize
from presto.data import frozen_partitions
from presto.training.output_contract import OutputConfiguration, build_output_contract
from presto.training.target_policy import affinity_target_policies

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
TAG = "effective-target-fit-contract-v1"
BASE = "https://github.com/pirl-unc/presto/releases/download/" + TAG + "/"
OLD = {
    "url": "https://github.com/pirl-unc/presto/releases/download/corrected-human-preflight-v1/prospective-supported-outputs.json",
    "sha256": "13fe8bd4dab158bd06ae31fc7996fba98f73b3944fa68c3f57463beec34e7cb0",
    "size": 3346,
}

OLD_RUN = {
    "url": "https://github.com/pirl-unc/presto/releases/download/corrected-human-preflight-v1/run.tar.gz",
    "sha256": "ce8d55cf1e7a14fb898f997918133f708526d145c65f01f97c37e2b0df6db0c8",
    "size": 7308268,
}


def pin(path):
    raw = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    assert presto.__version__ == "0.1.16"
    producer = json.loads((FAMILY / "reproduce/producer.json").read_text())
    assert producer["presto_version"] == presto.__version__
    # The caller imports the independently verified public wheel, not this .17 checkout.
    assert Path(presto.__file__).resolve().parent != REPO
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    cache = Path.home() / ".hitlist/presto-effective-target-fit/declaration"
    source = install_bundle(
        cache / "inputs",
        {"archived-supported-outputs.json": OLD, "old-preflight-run.tar.gz": OLD_RUN},
        download_options={"show_progress": False},
    )
    names = (
        "archived-supported-outputs.json",
        "prospective-supported-outputs.json",
        "declaration-change.json",
        "archived-membership-policy.json",
        "expected-membership-policy.json",
        "producer.json",
        "plan.md",
        "launch.json",
    )

    def builder(inputs, outputs):
        old = json.loads(Path(inputs["archived-supported-outputs.json"]).read_text())
        assert "affinity_target_policies" not in old["source_contract"]
        new = copy.deepcopy(old)
        policies = affinity_target_policies(
            build_output_contract(OutputConfiguration(**old["model_contract"]))
        )
        assert len(policies) == 9
        new["source_contract"]["affinity_target_policies"] = policies
        old_receipt = old["source_contract"]["frozen_membership"]["artifacts"]["policy.json"]
        with tarfile.open(inputs["old-preflight-run.tar.gz"]) as archive:
            member = archive.getmember("frozen_membership/policy.json")
            assert member.isfile()
            raw = archive.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest() == old_receipt["sha256"]
        assert len(raw) == old_receipt["size_bytes"]
        old_policy = json.loads(raw)
        assert old_policy["presto_version"] == "0.1.13"
        assert pin(frozen_partitions.__file__)["sha256"] == old_policy["recipe_sha256"]
        Path(outputs["archived-membership-policy.json"]).write_bytes(raw)
        new_policy = {**old_policy, "presto_version": "0.1.16"}
        write(outputs["expected-membership-policy.json"], new_policy)
        expected_receipt = pin(outputs["expected-membership-policy.json"])
        new["source_contract"]["frozen_membership"]["artifacts"]["policy.json"] = {
            "sha256": expected_receipt["sha256"],
            "size_bytes": expected_receipt["size"],
        }
        restored = copy.deepcopy(new)
        del restored["source_contract"]["affinity_target_policies"]
        restored["source_contract"]["frozen_membership"]["artifacts"]["policy.json"] = old_receipt
        assert restored == old
        shutil.copyfile(
            inputs["archived-supported-outputs.json"], outputs["archived-supported-outputs.json"]
        )
        write(outputs["prospective-supported-outputs.json"], new)
        for name in ("producer.json", "plan.md", "launch.json"):
            shutil.copyfile(inputs[name], outputs[name])
        write(
            outputs["declaration-change.json"],
            {
                "status": "new prospective family; original declaration immutable",
                "source_commit": commit,
                "dirty": False,
                "producer": producer,
                "script": pin(__file__),
                "old_manifest": OLD,
                "new_manifest": {
                    "url": BASE + "prospective-supported-outputs.json",
                    **pin(outputs["prospective-supported-outputs.json"]),
                },
                "added_paths": ["source_contract.affinity_target_policies"],
                "changed_paths": ["source_contract.frozen_membership.artifacts.policy.json"],
                "membership_receipt_change": {
                    "field": "presto_version",
                    "old": "0.1.13",
                    "new": "0.1.16",
                    "old_artifact": old_receipt,
                    "expected_artifact": new["source_contract"]["frozen_membership"]["artifacts"][
                        "policy.json"
                    ],
                    "partition_recipe_sha256": old_policy["recipe_sha256"],
                    "status": "prospective expected receipt; canonical full-input gate must reproduce it",
                },
                "unchanged": [
                    "model_contract",
                    "claims",
                    "dataset/supervision/evidence fingerprints",
                    "source/partition/parent graph fingerprints",
                ],
                "reason": "New family explicitly trains and compares in the historical effective target space; raw measurement diagnostics remain separate.",
                "scope": "A declaration is not a passed canonical full-input qualification or a fitted result.",
            },
        )

    sources = {
        "archived-supported-outputs.json": {
            "path": str(source["archived-supported-outputs.json"]),
            "sha256": OLD["sha256"],
        }
    }
    sources["old-preflight-run.tar.gz"] = {
        "path": str(source["old-preflight-run.tar.gz"]),
        "sha256": OLD_RUN["sha256"],
    }
    for name, path in {
        "producer.json": FAMILY / "reproduce/producer.json",
        "plan.md": REPO / "tasks/effective-target-fit-spec.md",
        "launch.json": FAMILY / "reproduce/launch.json",
    }.items():
        sources[name] = {"path": str(path), "sha256": pin(path)["sha256"]}
    outputs = materialize(
        cache / "generations" / commit,
        sources,
        transform={
            "version": "explicit-effective-target-declaration-v1",
            "options": {"producer": producer, "script": pin(__file__)},
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    for name in (
        "archived-supported-outputs.json",
        "prospective-supported-outputs.json",
        "declaration-change.json",
        "archived-membership-policy.json",
        "expected-membership-policy.json",
    ):
        shutil.copyfile(outputs[name], FAMILY / "reproduce" / name)
    inputs_path = FAMILY / "reproduce/frozen-inputs.json"
    inputs = json.loads(inputs_path.read_text())
    assert inputs["prospective-supported-outputs.json"] == OLD
    inputs["archived-supported-outputs.json"] = OLD
    inputs["prospective-supported-outputs.json"] = {
        "url": BASE + "prospective-supported-outputs.json",
        **pin(outputs["prospective-supported-outputs.json"]),
    }
    write(inputs_path, inputs)
    receipt = {
        "tag": TAG,
        "source_commit": commit,
        "files": {name: {"url": BASE + name, **pin(path)} for name, path in outputs.items()},
        "paths": {name: str(path) for name, path in outputs.items()},
    }
    write(FAMILY / "results/declaration-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
