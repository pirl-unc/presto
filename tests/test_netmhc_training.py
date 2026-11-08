"""Software fixtures for the source-faithful NetMHC BA projection."""

import hashlib

import pytest

from presto.data.modular import Observation
from presto.data.netmhc_training import (
    ArchiveBAIssue,
    invert_ba_target,
    parse_ba_line,
    peptide_cores,
    project_ba_rows,
)


def parse(line, *, archive="netmhcpan42", member="NetMHCpan_train/c000_ba", number=1):
    return parse_ba_line(
        line.encode(),
        archive=archive,
        member=member,
        fold=0,
        line_number=number,
    )


def resolver(identity):
    if identity == "UNRESOLVED":
        return None
    mhc_class = "II" if identity.startswith("DR") else "I"
    return {
        "mhc_class": mhc_class,
        "groove1": "ACDEFGHIKLM",
        "groove2": "NPQRSTVWYAC",
        "catalog_alleles": [f"HomoSapiens-{identity}"],
    }


def project(rows, **kwargs):
    return project_ba_rows(
        rows,
        resolve_identity=resolver,
        archive_hashes={"netmhcpan42": "a" * 64},
        source_names={"netmhcpan42": "NetMHCpan-4.2-training/BA"},
        development_peptides=kwargs.get("development_peptides", set()),
        development_cores=kwargs.get("development_cores", set()),
        train_pmhc=kwargs.get("train_pmhc", set()),
        validate_observation=Observation.from_dict,
    )


@pytest.mark.parametrize(
    "target,value,qualifier",
    [(0.0, 50000.0, ">="), (1.0, 1.0, "<="), (0.5, 50000**0.5, "=")],
)
def test_ba_target_inversion_preserves_censor_boundaries(target, value, qualifier):
    assert invert_ba_target(target) == pytest.approx((value, qualifier))


def test_parse_rejects_noncanonical_and_invalid_targets():
    assert isinstance(parse("ACDEXGHIK 0.5 HLA-A02:01 X"), ArchiveBAIssue)
    assert isinstance(parse("ACDEFGHIK 2 HLA-A02:01 X"), ArchiveBAIssue)
    assert parse("ACDEFGHIK 0.5 HLA-A02:01 X").target == 0.5


def test_projection_excludes_conflicts_duplicates_and_unresolved_rows():
    rows = [
        parse("ACDEFGHIK 0.5 HLA-A02:01 X", number=1),
        parse("ACDEFGHIK 0.5 HLA-A02:01 X", number=2),
        parse("LMNPQRSTV 0.2 HLA-A02:01 X", number=3),
        parse("LMNPQRSTV 0.3 HLA-A02:01 X", number=4),
        parse("RSTVWYACD 0.4 UNRESOLVED X", number=5),
    ]
    result = project(rows)
    assert len(result.observations) == 1
    assert sorted(row["disposition"] for row in result.dispositions) == sorted(
        [
            "admitted_train",
            "duplicate_exact_source_row",
            "conflicting_source_targets",
            "conflicting_source_targets",
            "nonhuman_or_unresolved_mhc",
        ]
    )
    assert len(result.conflicts) == len(result.duplicates) == 1
    observation = Observation.from_dict(result.observations[0])
    assert observation.assay_method is not None and not observation.assay_method.usable
    assert observation.provenance["assay_output"] == "unspecified_assay/IC50"
    assert len(observation.provenance["archive_lineage"]) == 2


def test_projection_applies_exact_core_and_train_pmhc_embargoes():
    exact = parse("ACDEFGHIK 0.5 HLA-A02:01 X", number=1)
    shared_core = parse("MACDEFGHIKL 0.5 HLA-A02:01 X", number=2)
    train_duplicate = parse("LMNPQRSTV 0.5 HLA-A02:01 X", number=3)
    train_key = (train_duplicate.peptide, "ACDEFGHIKLM", "NPQRSTVWYAC")
    result = project(
        [exact, shared_core, train_duplicate],
        development_peptides={exact.peptide},
        development_cores=set(peptide_cores(exact.peptide)),
        train_pmhc={train_key},
    )
    assert not result.observations
    assert {row["disposition"] for row in result.dispositions} == {
        "development_exact_peptide_embargo",
        "development_core9_embargo",
        "current_train_exact_pmhc_dedup",
    }


def test_observation_ids_and_disposition_order_are_input_order_independent():
    rows = [
        parse("ACDEFGHIK 0.5 HLA-A02:01 X", number=2),
        parse("LMNPQRSTV 0.4 HLA-A02:01 X", number=1),
    ]
    forward = project(rows)
    reverse = project(reversed(rows))
    assert forward == reverse
    assert all(
        row["observation_id"].startswith("netmhcpan42:ba:")
        and len(row["observation_id"].rsplit(":", 1)[1]) == hashlib.sha256().digest_size * 2
        for row in forward.observations
    )
