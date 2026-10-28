"""Independently check conservation, components, source bytes and permutation labels."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(row):
    return digest(json.dumps(row, sort_keys=True, allow_nan=False).encode())


def main():
    started = time.monotonic()
    pins = json.loads((F / "reproduce/data-pins.json").read_text())
    membership = {
        r["observation_id"]: r
        for r in csv.DictReader((F / "results/membership.csv").open())
    }
    expected, metadata, token_owner, parents, row_tokens = {}, {}, {}, [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    prior_ids, prior_peptides, prior_assays = set(), set(), set()
    independent_counts = Counter()
    for spec in pins["prior_source_files"]:
        path = Path(pins["prior_source_root"]) / spec["path"]
        assert digest(path.read_bytes()) == spec["sha256"]
        for line in path.open():
            row = json.loads(line)
            uid = row["observation_id"]
            assert uid not in prior_ids and row["split"] == "train"
            prior_ids.add(uid)
            prior_peptides.add(row["peptide"])
            provenance, mhc = row["provenance"], row["mhc"][0]
            prior_assays.add(provenance["original_measurement_id"])
            independent_counts[row["endpoint"], mhc["mhc_class"], row["qualifier"]] += 1
            tokens = {
                "peptide:" + row["peptide"],
                "measurement:" + provenance["original_measurement_id"],
            }
            tokens.update(
                "group:" + g
                for g in [row["group_id"], *provenance.get("group_aliases", [])]
            )
            i = len(parents)
            parents.append(i)
            row_tokens.append((uid, tokens))
            for token in tokens:
                if token in token_owner:
                    parents[root(i)] = root(token_owner[token])
                else:
                    token_owner[token] = i
            assigned = membership[uid]
            expected_row = dict(
                row,
                split=assigned["split"],
                group_id="assay-mechanism-component:" + assigned["component"],
            )
            expected_row["provenance"] = {
                **provenance,
                "prior_partition": "train",
                "prior_group_id": row["group_id"],
                "development_component": assigned["component"],
                "development_split_policy": pins["split_salt"],
            }
            expected[uid] = canonical(expected_row)
            groove = digest((mhc["groove1"] + "|" + mhc["groove2"]).encode())
            metadata[uid] = (
                row["endpoint"],
                mhc["mhc_class"],
                groove,
                row["qualifier"],
                row["unit"],
                row["value"],
                assigned["split"],
            )
    assert len(expected) == len(membership) == pins["prior_training_observations"]
    groups = defaultdict(list)
    for i, (uid, tokens) in enumerate(row_tokens):
        groups[root(i)].append((uid, tokens))
    for group in groups.values():
        minimum = min(token for _, tokens in group for token in tokens)
        component = digest((pins["split_salt"] + "|" + minimum).encode())
        bucket = int(component[:16], 16) / 2**64
        split = "train" if bucket < 0.8 else "validation" if bucket < 0.9 else "test"
        assert all(
            membership[uid]["component"] == component
            and membership[uid]["split"] == split
            for uid, _ in group
        )
    old = ROOT / "experiments" / pins["source_family"] / "reproduce/source-pins.json"
    old_pins = json.loads(old.read_text())
    old_heldout_rows = 0
    for spec in old_pins["input_sources"]:
        if spec["path"].endswith("-train.jsonl"):
            continue
        path = Path(pins["prior_source_root"]) / spec["path"]
        assert digest(path.read_bytes()) == spec["sha256"]
        for line in path.open():
            row = json.loads(line)
            assert row["observation_id"] not in prior_ids
            assert row["peptide"] not in prior_peptides
            assert row["provenance"]["original_measurement_id"] not in prior_assays
            old_heldout_rows += 1
    observed, recipients, donors, counts = set(), Counter(), Counter(), Counter()
    changed = 0
    for spec in pins["files"]:
        path = Path(pins["data_directory"]) / spec["path"]
        assert digest(path.read_bytes()) == spec["sha256"]
        n = 0
        for line in path.open():
            row = json.loads(line)
            uid = row["observation_id"]
            m = metadata[uid]
            assert row["split"] == spec["split"] == m[-1]
            assert row["endpoint"] == spec["endpoint"] == m[0]
            if spec["variant"] == "shuffled":
                note = row["provenance"].pop("diagnostic_label_permutation")
                donor = note["donor_observation_id"]
                assert row["split"] == "train" and row["endpoint"] != "ic50"
                assert (
                    note["kind"] == "synthetic_training_negative_control"
                    and note["seed"] == 73028
                )
                assert tuple(note["stratum"]) == m[:4] == metadata[donor][:4]
                assert m[4] == metadata[donor][4] and metadata[donor][-1] == "train"
                assert (
                    row["value"] == metadata[donor][5]
                    and note["original_value"] == m[5]
                )
                changed += row["value"] != m[5]
                row["value"] = m[5]
                recipients[uid] += 1
                donors[donor] += 1
            else:
                assert uid not in observed
                observed.add(uid)
                counts[row["split"], m[0], m[1], m[3]] += 1
            assert canonical(row) == expected[uid], uid
            n += 1
        assert n == spec["rows"]
    assert observed == set(expected)
    assert donors == recipients and all(n == 1 for n in recipients.values())
    assert (
        sum(recipients.values()) == pins["shuffled_rows"]
        and changed == pins["changed_shuffled_rows"]
    )
    assert counts == Counter(
        {
            (r["split"], r["endpoint"], r["mhc_class"], r["qualifier"]): r["n"]
            for r in pins["counts"]
        }
    )
    inventory = json.loads((F / "results/original-training-inventory.json").read_text())
    assert independent_counts == Counter(
        {
            (r["endpoint"], r["mhc_class"], r["qualifier"]): r["n"]
            for r in inventory["by_endpoint_class_qualifier"]
        }
    )
    result = dict(
        status="passed",
        original_rows=len(expected),
        independently_rebuilt_components=len(groups),
        old_heldout_rows_excluded=old_heldout_rows,
        measured_rows_checked=len(observed),
        synthetic_rows_checked=len(recipients),
        actually_changed=changed,
        source_bytes_and_full_original_rows_conserved=True,
        target_independent_split=True,
        permutation_is_within_training_endpoint_class_groove_qualifier=True,
        seconds=time.monotonic() - started,
    )
    (F / "results/data-independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
