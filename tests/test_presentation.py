from presto.data.presentation import (
    Protein,
    RelatedSequences,
    background_candidate,
    map_peptides,
    protein_families,
    source_junction,
)


def test_related_sequence_rules_cover_long_class_ii_and_short_class_i():
    protected = RelatedSequences(['ACDEFGHIKLMNPQRST', 'ACDEFG'])
    assert protected.reason('ACDEFGHIKLMNPQRST') == 'exact'
    assert protected.reason('VVVGHIKLMNPVVVV') == 'shared8mer'
    assert protected.reason('ACDEYG') == 'hamming1'
    assert protected.reason('YYYYYYYYY') is None


def test_every_source_occurrence_joins_the_protein_family_and_masks_disagreement():
    proteins = {
        'a': Protein('a', 'g1', 'VVACDEFGHIKLMN'),
        'b': Protein('b', 'g1', 'WWACDEFGHIKLMN'),
        'c': Protein('c', 'g2', 'WWPQRSTVWYYYYY'),
    }
    hits = map_peptides(['ACDEFGHI', 'WW'], proteins)
    assert hits['ACDEFGHI'] == [('a', 2), ('b', 2)]
    assert len(set(protein_families(hits).values())) == 1
    context, policy = source_junction('ACDEFGHI', hits['ACDEFGHI'], proteins)
    assert policy == 'ambiguous_reference_junction'
    assert all(value is None for value in context.values())


def test_background_uses_possible_source_same_length_and_excludes_known_sequences():
    sequence = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIK'
    proteins = {'p': Protein('p', 'g', sequence)}
    known = {'ACDEFGHIK', 'CDEFGHIKL'}
    held = RelatedSequences(['NPQRSTVWY'])
    args = dict(observed=known, protected=held, key='measurement')
    actual = background_candidate('ACDEFGHIK', [('p', 0)], proteins, **args)
    assert actual == background_candidate('ACDEFGHIK', [('p', 0)], proteins, **args)
    peptide, identifier, start, _ = actual
    assert identifier == 'p' and len(peptide) == 9
    assert sequence[start : start + 9] == peptide
    assert peptide not in known and held.reason(peptide) is None


def test_noncanonical_reference_flank_is_unknown_without_erasing_the_other_flank():
    proteins = {'p': Protein('p', 'g', 'UACDEFGHIKLMN')}
    context, policy = source_junction('ACDEFGHI', [('p', 1)], proteins)
    assert context == {
        'source_n': None,
        'source_c': 'KLMN',
        'n_terminus': False,
        'c_terminus': False,
    }
    assert policy == 'unanimous_reference_junction'
