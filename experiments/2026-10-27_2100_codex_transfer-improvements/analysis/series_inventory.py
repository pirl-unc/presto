"""Training-only inventory of candidate measured series, never a label adapter."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


def main():
    pins = json.loads((FAMILY / 'reproduce/source-pins.json').read_text())
    path = ROOT / 'artifacts' / FAMILY.name / 'upload/inputs/joint-train.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pins['joint_input_hashes']['train']
    fields = (
        'pmid',
        'assay_iri',
        'submission_id',
        'assay_method',
        'reference_title',
        'supplementary_file',
    )
    present = Counter()
    groups = defaultdict(set)
    assays = defaultdict(set)
    counts = Counter()
    rows = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['split'] == 'train' and row['qualifier'] == '='
            rows += 1
            raw = row['provenance']['raw_record']
            kind = row['mhc'][0]['mhc_class']
            counts[kind] += 1
            for field in fields:
                if raw.get(field):
                    present[field] += 1
            mhc = row['mhc'][0]
            groove = hashlib.sha256((mhc['groove1'] + '|' + mhc['groove2']).encode()).hexdigest()
            if raw.get('assay_iri'):
                assays[(groove, str(raw['assay_iri']))].add(row['peptide'])
            if raw.get('pmid') and raw.get('assay_method'):
                key = (
                    kind,
                    groove,
                    str(raw['pmid']),
                    str(raw['assay_method']),
                    str(raw.get('measurement_units')),
                )
                groups[key].add(row['peptide'])
    edges = Counter()
    families = Counter()
    for key, peptides in groups.items():
        pairs = set()
        for peptide in peptides:
            for total_trim in range(1, 5):
                for left in range(total_trim + 1):
                    right = total_trim - left
                    shorter = peptide[left : len(peptide) - right] if right else peptide[left:]
                    if len(shorter) >= 8 and shorter in peptides:
                        pairs.add((peptide, shorter))
        if pairs:
            families[key[0]] += 1
            edges[key[0]] += len(pairs)
    result = dict(
        scope=(
            'descriptive training-only provenance inventory; no new labels, '
            'fitting, or held-out access'
        ),
        source_sha256=pins['joint_input_hashes']['train'],
        training_rows=rows,
        classes=dict(counts),
        field_present_rows=dict(present),
        same_mhc_assay_identifiers_with_multiple_peptides=sum(len(v) > 1 for v in assays.values()),
        candidate_same_mhc_publication_method_groups_with_nested_peptides=dict(families),
        candidate_terminally_nested_sequence_pairs=dict(edges),
        candidate_definition=(
            'measured exact peptides in the same MHC/publication/method/unit group; '
            'total terminal trimming1..4; shorter length>=8'
        ),
        admission_decision=(
            'none: publication/method agreement does not verify matched assay '
            'conditions or plate/panel membership; no rank labels derived'
        ),
        next_requirement=(
            'curated experimental-series/condition membership and family-disjoint '
            'splits before a measured-series ranking experiment'
        ),
    )
    out = FAMILY / 'results/training-series-inventory.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
