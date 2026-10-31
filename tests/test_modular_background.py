"""Synthetic comparison labels preserve lineage and cannot leak across splits."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
from presto.data.modular import Observation
from presto.data.modular_background import LigandBackgroundSampler
from presto.data.modular_hitlist import HumanMHCCatalog
from presto.tests.test_modular_hitlist import reference


def positive(peptide='ACDEFGHI', split='test'):
    pair = HumanMHCCatalog([reference()]).resolve('HLA-A*02:01', 'I')
    return Observation(
        'positive',
        'fixture',
        'elution_observation',
        peptide,
        (pair,),
        value=1.0,
        unit='binary',
        split=split,
        group_id='paper',
        provenance={'study_id': 'paper'},
    )


def test_background_is_length_matched_source_scoped_and_excludes_known_positives():
    proteins = {'source': 'ACDEFGHIKLMNPQRSTVWY', 'unrelated': 'A' * 30}
    proteome = SimpleNamespace(proteins=proteins, lookup=lambda _: [('source', 0)])
    forbidden = {'ACDEFGHI', 'CDEFGHIK'}
    sampler = LigandBackgroundSampler(
        proteome,
        seed=42,
        reference='fixture:v1',
        positive_peptides=forbidden,
        peptide_splits={'DEFGHIKL': 'train'},
    )
    row = sampler.sample(positive())
    assert len(row.peptide) == 8 and row.peptide not in forbidden | {'DEFGHIKL'}
    assert row.peptide in proteins['source']
    assert row.provenance['protein_id'] == 'source'
    assert row.provenance['background_kind'] == 'same_protein'
    assert row.provenance['measured_outcome'] is False
    assert row.provenance['parent_observation_id'] == 'positive'
    assert row.value == 0 and row.split == 'test' and row.group_id == 'paper'
    assert sampler.sample(positive()).to_dict() == row.to_dict()


def test_unmapped_fallback_and_true_exhaustion_are_explicit():
    proteome = SimpleNamespace(proteins={'p': 'KLMNPQRS'}, lookup=lambda _: [])
    sampler = LigandBackgroundSampler(
        proteome, seed=0, reference='fixture:v1', positive_peptides={'ACDEFGHI'}, peptide_splits={}
    )
    row = sampler.sample(positive())
    assert row.provenance['background_kind'] == 'reference_proteome'
    assert row.peptide == 'KLMNPQRS'
    with pytest.raises(ValueError, match='exhausted'):
        sampler.sample(replace(positive(), observation_id='other', split='train'))
    with pytest.raises(ValueError, match='partitioned positive'):
        sampler.sample(replace(positive(), value=0))
