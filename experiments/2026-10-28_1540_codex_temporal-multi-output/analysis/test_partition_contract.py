"""Artificial metadata-only selection invariants, independent of biological targets."""

from collections import Counter
from pathlib import Path
import sqlite3
import json
import sys

C = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(C))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition import Components, census_components, select  # noqa: E402
from prepare_worker import weights  # noqa: E402


def test_connected_aliases_are_transitive_and_stable():
    c = Components()
    c.union('c', 'b')
    c.union('a', 'd')
    c.union('d', 'c')
    assert c.groups() == {'a': ['a', 'b', 'c', 'd']}


def test_selection_reserves_training_and_validation_support():
    studies = {
        str(i): dict(
            id=str(i),
            aliases=['pmid:' + str(i)],
            years=[2026 if i == 0 else 2020],
            n=100,
            counts=Counter({('ic50', 'I'): 100}),
        )
        for i in range(5)
    }
    assignments, receipt = select(studies, test_studies=('pmid:0',))
    assert assignments['0'] == 'test'
    assert sum(v == 'validation' for v in assignments.values()) == 1
    assert sum(v == 'train' for v in assignments.values()) == 3
    assert receipt['missing_2026_studies'] == []
    # No prediction or physical target values are an argument to the selector.
    again, _ = select(dict(reversed(list(studies.items()))), test_studies=('pmid:0',))
    assert assignments == again


def test_single_study_output_is_not_stolen_from_training():
    studies = {
        str(i): dict(
            id=str(i), aliases=[str(i)], years=[2020], n=100, counts=Counter({('kon', 'I'): 100})
        )
        for i in range(1)
    }
    assignments, receipt = select(studies, test_studies=())
    assert assignments == {'0': 'train'}
    assert receipt['trace'] == []


def test_reported_negative_coverage_requires_both_labels_in_same_study(tmp_path):
    db = sqlite3.connect(':memory:')
    db.executescript(
        'CREATE TABLE rows(id,study,provenance_id,endpoint,mhc_class,excluded,data); '
        'CREATE TABLE aliases(id,study); '
        'CREATE TABLE source_studies(provenance_id,study);'
    )
    for i, (study, label) in enumerate(
        [('pmid:a', 1), ('pmid:b', 0), ('pmid:c', 1), ('pmid:c', 0)]
    ):
        db.execute(
            'INSERT INTO rows VALUES (?,?,?,?,?,?,?)',
            (
                str(i),
                study,
                'p' + str(i),
                'elution_observation',
                'I',
                0,
                json.dumps({'qualifier': '=', 'value': label}),
            ),
        )
        db.execute('INSERT INTO aliases VALUES (?,?)', (str(i), study))
    # Link A and B by contributor; aggregation must not invent within-study labels.
    db.execute('INSERT INTO source_studies VALUES (?,?)', ('p0', 'pmid:b'))
    refs = tmp_path / 'refs.jsonl'
    refs.write_text(
        ''.join(
            json.dumps({'metadata': {'Reference | PMID': s, 'Reference | Date': '2020'}}) + '\n'
            for s in 'abc'
        )
    )
    c, studies = census_components(db, refs)
    assert ('reported_ms_outcomes', 'I') not in studies[c.find('pmid:a')]['counts']
    assert studies[c.find('pmid:c')]['counts'][('reported_ms_outcomes', 'I')] == 1


def test_sampling_frequencies_protect_primary_labels_and_soft_balance_classes():
    w = weights(
        {
            ('elution_observation', 'I'): 900,
            ('elution_observation', 'II'): 100,
            ('ic50', 'I'): 100,
            ('kd', 'I'): 100,
            ('half_life', 'II'): 25,
            ('qualitative_binding', 'I'): 100,
        }
    )
    assert abs(w[('elution_observation', 'I')] + w[('elution_observation', 'II')] - 0.6) < 1e-12
    assert abs(w[('ic50', 'I')] - 0.2) < 1e-12
    assert abs(w[('kd', 'I')] / w[('half_life', 'II')] - 2) < 1e-12
    assert abs(w[('elution_observation', 'I')] / w[('elution_observation', 'II')] - 3) < 1e-12


def test_preparation_through_real_common_index_and_preflight(tmp_path):
    import gzip
    import presto
    from presto.data.modular import Observation, MHCInput
    from presto.data.modular_sources import ObservationIndex
    from prepare_worker import prepare, sha

    root = tmp_path / 'corpus'
    bundle = tmp_path / 'bundle'
    (root / 'admission').mkdir(parents=True)
    (root / 'source').mkdir()
    bundle.mkdir()
    db = sqlite3.connect(root / 'admission/admitted.sqlite')
    db.executescript(
        'CREATE TABLE rows(id,peptide,study,provenance_id,endpoint,mhc_class,excluded,data); '
        'CREATE TABLE aliases(id,study); '
        'CREATE TABLE source_studies(provenance_id,study); '
        'CREATE TABLE positive_peptides(peptide PRIMARY KEY);'
    )
    references = []
    all_peptides = []
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    for s in range(8):
        pmid = '41534530' if s == 0 else str(s)
        study = 'pmid:' + pmid
        references.append(
            {
                'metadata': {
                    'Reference | PMID': pmid,
                    'Reference | Date': '2026' if s == 0 else '2020',
                }
            }
        )
        for n in range(24):
            i = s * 24 + n
            peptide = 'ACDEF' + aa[i // len(aa)] + aa[i % len(aa)] + 'GHIKL'
            all_peptides.append(peptide)
            endpoint = 'ic50' if n < 12 else 'elution_observation'
            cls = 'I' if n % 2 else 'II'
            value = float(1 + n) if n < 12 else float(n % 3 != 0)
            row = Observation(
                str(i),
                'fixture',
                endpoint,
                peptide,
                (MHCInput('ACDE' * 20, 'FGHI' * 20, cls),),
                value=value,
                unit='nM' if n < 12 else 'binary',
                provenance={
                    'study_id': study,
                    'source_contributor_id': str(i),
                    'measurement_namespace': 'fixture',
                    'original_measurement_id': str(i),
                    'measured_outcome': True,
                    'label_kind': 'reported_positive' if value else 'reported_nondetection',
                    'raw_record': {
                        'pmid': pmid,
                        'assay_method': 'fixture',
                        'quantitative_value': value,
                        'measurement_units': 'nM',
                        'measurement_inequality': '=',
                        'is_ms_observation': bool(value),
                        'ms_outcome_status': 'positive_result' if value else 'negative',
                        'qualitative_measurement': 'Positive' if value else 'Negative',
                    },
                },
            )
            db.execute(
                'INSERT INTO rows VALUES (?,?,?,?,?,?,?,?)',
                (str(i), peptide, study, str(i), endpoint, cls, 0, json.dumps(row.to_dict())),
            )
            db.execute('INSERT INTO aliases VALUES (?,?)', (str(i), study))
            if endpoint == 'elution_observation' and value:
                db.execute('INSERT INTO positive_peptides VALUES (?)', (peptide,))
    db.commit()
    db.close()
    (root / 'source/references.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in references))
    with gzip.open(bundle / 'human-ensembl112.pep.fa.gz', 'wt') as stream:
        stream.write('>fixture-protein\n' + 'MNPQRSTVWY'.join(all_peptides) + '\n')
    runtime = Path(presto.__file__).parent
    pins = {
        'files': {'data/modular.py': sha(runtime / 'data/modular.py')},
        'proteome': {
            'sha256': sha(bundle / 'human-ensembl112.pep.fa.gz'),
            'source': 'artificial test protein',
        },
    }
    (bundle / 'runtime-pins.json').write_text(json.dumps(pins))
    result = prepare(root, lambda *args: None, bundle=bundle, runtime=runtime)
    assert result['status'] == 'prepared'
    index = ObservationIndex(result['manifests']['wide-multi-s42']['path'])
    assert set(index.partitions) == {'train:assay', 'validation:assay', 'test:assay'}
    assert result['model']['endpoints'] == ['elution_observation', 'ic50']
    assert result['background_contract']['train_ratio'] == 1
    assert result['background_contract']['evaluation_ratio'] == 5
    for split in ('train', 'validation', 'test'):
        rows = [index[i] for i in index.partitions[split + ':assay']]
        positive = sum(r.endpoint == 'elution_observation' and r.value == 1 for r in rows)
        background = sum(r.source == 'hitlist_ligand_background' for r in rows)
        assert background == positive * (1 if split == 'train' else 5)
    from verify_cohort import verify

    assert verify(root)['status'] == 'passed'


def test_ligand_report_does_not_mix_synthetic_and_reported_negatives():
    from worker import ligand_reports

    def row(identifier, source, value, prediction, provenance):
        return dict(
            observation_id=identifier,
            endpoint='elution_observation',
            source=source,
            target=value,
            prediction=prediction,
            loss=0.1,
            mhc_classes=['I'],
            peptide='ACDEFGHIKL',
            observation={'provenance': {'study_id': 'fixture-study', **provenance}},
        )

    positive = row('p', 'hitlist_native_ms', 1, 2, {})
    negative = row('n', 'hitlist_native_ms', 0, 3, {'label_kind': 'reported_nondetection'})
    background = row(
        'b',
        'hitlist_ligand_background',
        0,
        -1,
        {'parent_observation_id': 'p', 'background_kind': 'same_protein'},
    )
    report = ligand_reports([positive, negative, background])
    assert report['backgrounds']['same_protein']['metrics']['elution_observation']['auroc'] == 1.0
    assert report['backgrounds']['same_protein']['length_only']['auroc'] == 0.5
    assert report['source_reported_by_study'][0]['metrics']['elution_observation']['auroc'] == 0.0
    assert 'reference_proteome' not in report['backgrounds']


def test_vocabulary_column_projection_preserves_condition_scope():
    from presto.data.modular_hitlist_ms import FACTOR_FIELDS, study_factor_vocabulary

    fields = {
        'peptide',
        'pmid',
        'sample_attribution',
        'condition_id',
        'condition_status',
        'condition_evidence',
        'condition_reference',
        *FACTOR_FIELDS,
    }
    base = dict(
        pmid=123,
        sample_attribution='curated_sample_label',
        condition_id='arm1',
        condition_status='annotated',
        condition_evidence='primary_source',
        condition_reference='fixture',
        irrelevant_outcome=0,
        irrelevant_description='not a modeling input',
    )
    rows = [dict(base, **{field: 'fixture-' + field}) for field in FACTOR_FIELDS]
    rows += [dict(base, **{field: 'none'}) for field in FACTOR_FIELDS]
    rows += [dict(base, condition_id='', condition_drugs='unresolved-arm-token')]
    projected = [{k: v for k, v in row.items() if k in fields} for row in rows]
    assert study_factor_vocabulary(rows) == study_factor_vocabulary(projected)
    assert study_factor_vocabulary(projected)['123']['condition_drugs'] == (
        'fixture-condition_drugs',
    )
