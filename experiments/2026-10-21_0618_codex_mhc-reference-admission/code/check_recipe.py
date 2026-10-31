"""Artificial source-contract checks; no real cohort or biological predictions."""

from copy import deepcopy
from pathlib import Path
import runpy

recipe = runpy.run_path(str(Path(__file__).with_name("launch.py")), run_name="recipe_checks")


def source(index, taxon, mhc_class, chain, first="", second=""):
    return {
        "source": "software",
        "source_id": str(index),
        "two_field_allele": f"fixture-{index}",
        "gene": "software",
        "species": taxon,
        "mhc_class": mhc_class,
        "chain": chain,
        "groove1": first,
        "groove2": second,
        "groove1_len": str(len(first)),
        "groove2_len": str(len(second)),
        "sequence": "M" + first + second,
        "groove_status": "ok",
        "groove_flags": "",
        "is_functional": "True",
        "is_null": "False",
        "is_questionable": "False",
        "is_pseudogene": "False",
    }


rows = [
    source(0, "Homo sapiens", "I", "alpha", "A" * 80, "C" * 90),
    source(1, "Homo sapiens", "I", "alpha", "A" * 80, "D" * 90),
    source(2, "Homo sapiens", "II", "alpha", "A" * 80),
    source(3, "Mus musculus (Mouse)", "II", "beta", second="G" * 90),
    source(4, "Rattus norvegicus (Norway rat)", "I", "alpha", "E" * 80, "F" * 90),
    source(5, "Macaca mulatta", "II", "alpha", "H" * 80),
]
observations, dispositions, annotations, conflicts = recipe["prepare"](rows)
assert not conflicts and len(observations) == 7
assert sum(len(row["aliases"]) for row in annotations) == 9
assert len({row.group_id for row in observations}) == 4
assert all(bool(row.mhc[0].groove1) != bool(row.mhc[0].groove2) for row in observations)
assert all(set(row.identity_labels) <= {"mhc1", "mhc2"} for row in observations)
assert dispositions[2]["status"] == "duplicate_alias"
assert len(dispositions[1]["alias_annotations"]) == 1
# A taxonomic ambiguity defers every parent containing that input; no majority vote.
rows.append(source(6, "Mus musculus (Mouse)", "II", "alpha", "A" * 80))
observations, dispositions, _, conflicts = recipe["prepare"](rows)
assert len(observations) == 4 and len(conflicts) == 1
assert [i for i, row in enumerate(dispositions) if row["status"] == "deferred"] == [0, 1, 2, 6]
# Mechanical guards must distinguish source anomalies from trustworthy labels.
for field, value, reason in (
    ("chain", "B2M", "unsupported_class_chain"),
    ("species", "Mus sp.", "taxon_outside_frozen_vocabulary"),
    ("groove_flags", "non_groove_gene", "unsupported_groove_status_or_flags"),
    ("groove1_len", "79", "declared_domain_length_mismatch"),
    ("sequence", "M" * 180, "domain_absent_from_source_protein"),
):
    row = deepcopy(rows[0])
    row[field] = value
    assert recipe["domains"](row)[0] == reason
print(
    "Passed: source slots, alias collapse, parent components, "
    "species conflict deferral and five mechanical guards"
)
