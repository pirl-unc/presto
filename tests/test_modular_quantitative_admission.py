"""Literal assay identity and strict common source admission."""

from dataclasses import asdict
import pytest
from presto.data.modular_hitlist import (
    AdmissionError,
    HumanMHCCatalog,
    purified_assay_observation,
    purified_ic50_observation,
)
from presto.tests.test_modular_hitlist import raw, reference


@pytest.mark.parametrize(
    'response,unit,endpoint',
    [
        ('half maximal inhibitory concentration (IC50)', 'nM', 'ic50'),
        ('dissociation constant KD', 'nM', 'kd'),
        ('half maximal effective concentration (EC50)', 'nM', 'ec50'),
        ('dissociation constant KD (~IC50)', 'nM', 'kd_via_ic50'),
        ('dissociation constant KD (~EC50)', 'nM', 'kd_via_ec50'),
        ('half life', 'min', 'half_life'),
        ('off rate', '1/s', 'koff'),
    ],
)
def test_assay_identity_and_original_bound_survive(response, unit, endpoint):
    source = raw(
        response_measured=response,
        measurement_units=unit,
        measurement_inequality='>',
        quantitative_measurement='>10',
    )
    row = purified_assay_observation(
        source, HumanMHCCatalog([reference()]), source_sha256='0' * 64, ordinal=4
    )
    assert (row.endpoint, row.unit, row.qualifier, row.value) == (endpoint, unit, '>', 10)
    assert row.provenance['raw_record'] == source


@pytest.mark.parametrize(
    'response,unit',
    [
        ('50% dissociation temperature', 'C'),
        ('on rate', 'nM^-1s^-1'),
        ('half life', 'nM'),
        ('off rate', 'min'),
    ],
)
def test_unsupported_assay_or_units_are_not_reinterpreted(response, unit):
    with pytest.raises(AdmissionError):
        purified_assay_observation(
            raw(response_measured=response, measurement_units=unit),
            HumanMHCCatalog([reference()]),
            source_sha256='0' * 64,
            ordinal=4,
        )


def test_existing_ic50_contract_is_unchanged():
    source = raw()
    catalog = HumanMHCCatalog([reference()])
    kw = dict(source_sha256='0' * 64, ordinal=4)
    assert asdict(purified_ic50_observation(source, catalog, **kw)) == asdict(
        purified_assay_observation(source, catalog, **kw)
    )
    with pytest.raises(AdmissionError, match='outside_direct_iedb_cedar_ic50'):
        purified_ic50_observation(raw(response_measured='dissociation constant KD'), catalog, **kw)
