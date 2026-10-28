"""Inventory and repartition only earlier training rows, preserving source identity."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
OLD = ROOT / "experiments/2026-10-27_2255_codex_shared-multi-assay-scale"
INPUT = ROOT / "artifacts" / OLD.name / "upload/inputs"
DATA = ROOT / "data/modular_cohorts/2026-10-28-assay-transfer-mechanisms"
SALT = "2026-10-28-assay-transfer-mechanisms-v1"
ENDPOINTS = ("ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50", "half_life")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair(row):
    m = row["mhc"][0]
    return hashlib.sha256((m["groove1"] + "|" + m["groove2"]).encode()).hexdigest()


def component_tokens(row):
    p = row["provenance"]
    return [
        "peptide:" + row["peptide"],
        "measurement:" + p["original_measurement_id"],
        *["group:" + value for value in [row["group_id"], *p.get("group_aliases", [])]],
    ]


class Components:
    def __init__(self):
        self.parents = {}

    def find(self, value):
        parent = self.parents.setdefault(value, value)
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, a, b):
        x, y = sorted((self.find(a), self.find(b)))
        self.parents[y] = x


def main():
    started = time.monotonic()
    assert not (F / "reproduce/data-pins.json").exists(), "already frozen"
    DATA.mkdir(parents=True, exist_ok=True)
    pins = json.loads((OLD / "reproduce/source-pins.json").read_text())
    specs = [s for s in pins["input_sources"] if s["path"].endswith("-train.jsonl")]
    rows = []
    for spec in specs:
        path = INPUT / spec["path"]
        assert sha(path) == spec["sha256"]
        for line in path.open():
            row = json.loads(line)
            assert row["split"] == "train" and row["endpoint"] in ENDPOINTS
            rows.append(row)
    assert len(rows) == 134561
    assert len({r["observation_id"] for r in rows}) == len(rows)
    counts, lengths, genes, sources, species = (Counter() for _ in range(5))
    allele_rows, aliases, allele_peptides, contexts = (
        defaultdict(Counter),
        defaultdict(set),
        defaultdict(set),
        set(),
    )
    components = Components()
    for row in rows:
        m, p = row["mhc"][0], row["provenance"]
        key, cls = pair(row), m["mhc_class"]
        counts[(row["endpoint"], cls, row["qualifier"])] += 1
        lengths[(cls, len(row["peptide"]))] += 1
        species[(m["species1"], m["species2"])] += 1
        sources[(row["source"], p["raw_record"].get("reference_iri", "missing"))] += 1
        names = m["provenance"].get("catalog_alleles", [])
        aliases[key].add(" / ".join(names))
        genes[(cls, " / ".join(n.split("*")[0] for n in names))] += 1
        allele_rows[key][(row["endpoint"], cls, row["qualifier"])] += 1
        allele_peptides[key].add(row["peptide"])
        contexts.add((row["peptide"], key))
        tokens = component_tokens(row)
        for token in tokens[1:]:
            components.union(tokens[0], token)
    inventory = dict(
        observations=len(rows),
        unique_peptides=len({r["peptide"] for r in rows}),
        unique_pmhc_contexts=len(contexts),
        groove_pairs=len(allele_rows),
        by_endpoint_class_qualifier=[
            dict(endpoint=e, mhc_class=c, qualifier=q, n=n)
            for (e, c, q), n in sorted(counts.items())
        ],
        species=[dict(partner1=a, partner2=b, n=n) for (a, b), n in species.items()],
        genes=[dict(mhc_class=c, genes=g, n=n) for (c, g), n in genes.most_common()],
        lengths=[
            dict(mhc_class=c, length=l, n=n) for (c, l), n in sorted(lengths.items())
        ],
        source_study=[
            dict(source=s, study=r, n=n) for (s, r), n in sources.most_common()
        ],
        original_admission=json.loads(
            (
                ROOT
                / "experiments/2026-10-27_2235_codex_shared-support-audit/results/data-admission.json"
            ).read_text()
        )["counts"],
    )
    allele_table = []
    for key, cc in allele_rows.items():
        classes = {c for _, c, _ in cc}
        assert len(classes) == 1
        allele_table.append(
            dict(
                groove_id=key,
                mhc_class=next(iter(classes)),
                catalog_names="; ".join(sorted(aliases[key])),
                observations=sum(cc.values()),
                unique_peptides=len(allele_peptides[key]),
                exact=sum(n for (_, _, q), n in cc.items() if q == "="),
                **{
                    e: sum(n for (ep, _, _), n in cc.items() if ep == e)
                    for e in ENDPOINTS
                },
            )
        )
    allele_table.sort(key=lambda r: -r["observations"])
    table(F / "results/original-training-alleles.csv", allele_table)
    inventory["allele_concentration"] = {
        cls: {
            "observations": sum(
                r["observations"] for r in allele_table if r["mhc_class"] == cls
            ),
            "groove_pairs": sum(r["mhc_class"] == cls for r in allele_table),
            "top10": [r for r in allele_table if r["mhc_class"] == cls][:10],
        }
        for cls in ("I", "II")
    }
    write(F / "results/original-training-inventory.json", inventory)
    membership = []
    component_sizes = Counter()
    split_counts = Counter()
    rewritten = []
    for row in rows:
        root = components.find("peptide:" + row["peptide"])
        digest = hashlib.sha256((SALT + "|" + root).encode()).hexdigest()
        bucket = int(digest[:16], 16) / 2**64
        split = "train" if bucket < 0.8 else "validation" if bucket < 0.9 else "test"
        membership.append(
            dict(observation_id=row["observation_id"], component=digest, split=split)
        )
        component_sizes[digest] += 1
        split_counts[
            (split, row["endpoint"], row["mhc"][0]["mhc_class"], row["qualifier"])
        ] += 1
        new = dict(row, split=split, group_id="assay-mechanism-component:" + digest)
        new["provenance"] = {
            **row["provenance"],
            "prior_partition": "train",
            "prior_group_id": row["group_id"],
            "development_component": digest,
            "development_split_policy": SALT,
        }
        rewritten.append(new)
    table(F / "results/membership.csv", membership)
    original_by_id = {r["observation_id"]: r for r in rewritten}
    shuffled = {}
    strata = defaultdict(list)
    for row in rewritten:
        if row["split"] == "train" and row["endpoint"] != "ic50":
            strata[
                (
                    row["endpoint"],
                    row["mhc"][0]["mhc_class"],
                    pair(row),
                    row["qualifier"],
                )
            ].append(row)
    rng = random.Random(73028)
    permutation = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r["observation_id"])
        donors = list(group)
        rng.shuffle(donors)
        for row, donor in zip(group, donors, strict=True):
            assert row["unit"] == donor["unit"]
            new = dict(row, value=donor["value"])
            new["provenance"] = {
                **row["provenance"],
                "diagnostic_label_permutation": {
                    "kind": "synthetic_training_negative_control",
                    "seed": 73028,
                    "original_value": row["value"],
                    "donor_observation_id": donor["observation_id"],
                    "stratum": list(key),
                },
            }
            shuffled[row["observation_id"]] = new
            permutation.append(
                dict(
                    observation_id=row["observation_id"],
                    donor_id=donor["observation_id"],
                    changed=row["value"] != donor["value"],
                )
            )
    table(F / "results/permutation-membership.csv", permutation)
    output_specs = []
    for split in ("train", "validation", "test"):
        for endpoint in ENDPOINTS:
            selected = [
                r for r in rewritten if (r["split"], r["endpoint"]) == (split, endpoint)
            ]
            if not selected:
                continue
            variants = (
                ("measured", "shuffled")
                if split == "train" and endpoint != "ic50"
                else ("measured",)
            )
            for variant in variants:
                path = DATA / f"{endpoint}-{split}-{variant}.jsonl"
                with path.open("x") as stream:
                    for row in selected:
                        value = (
                            shuffled[row["observation_id"]]
                            if variant == "shuffled"
                            else row
                        )
                        stream.write(
                            json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
                        )
                output_specs.append(
                    dict(
                        path=path.name,
                        sha256=sha(path),
                        split=split,
                        endpoint=endpoint,
                        variant=variant,
                        rows=len(selected),
                    )
                )
    result = dict(
        status="prepared",
        source_family=OLD.name,
        prior_training_observations=len(rows),
        prior_source_files=specs,
        prior_source_root=str(INPUT),
        split_salt=SALT,
        component_count=len(component_sizes),
        largest_components=component_sizes.most_common(10),
        counts=[
            dict(split=s, endpoint=e, mhc_class=c, qualifier=q, n=n)
            for (s, e, c, q), n in sorted(split_counts.items())
        ],
        shuffled_rows=len(permutation),
        changed_shuffled_rows=sum(r["changed"] for r in permutation),
        membership_sha256=sha(F / "results/membership.csv"),
        permutation_sha256=sha(F / "results/permutation-membership.csv"),
        data_directory=str(DATA),
        files=output_specs,
        seconds=time.monotonic() - started,
        scope="New from-scratch development split of previously used training rows; original heldouts excluded; no external qualification.",
    )
    assert len(original_by_id) == len(rows)
    write(F / "reproduce/data-pins.json", result)
    write(DATA / "preparation.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("prior_source_files", "files")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
