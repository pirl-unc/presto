"""Recover source-supported forms by intersecting accepted ligands with original PSMs.

The Spectrum Mill export declares terminal groups, cysteine chemistry and variable
modifications separately. A blank modifications cell alone never means unmodified.
This audit limits recovery to the exact published ligand/allele set, retains every
supporting file and row, and does not claim absence of other forms from an incomplete
acquisition. Source rows are observations of known forms, not edits to unknown rows.
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import re
import time

F = Path(__file__).resolve().parents[1]
ROOT = Path('/Users/iskander/code/presto/artifacts') / F.name / 'recovered'
AA = frozenset('ACDEFGHIKLMNPQRSTVWY')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def chemistry(row):
    fields = ('sequence', 'modifications', 'variableSites', 'nterm', 'cterm', 'cys')
    if any(k not in row or row[k] is None for k in fields):
        return 'unknown'
    # Exact export declarations: Hydrogen/Free Acid denote native terminal groups.
    if (
        row['nterm'] == 'Hydrogen'
        and row['cterm'] == 'Free Acid'
        and not any(row[k] for k in ('modifications', 'variableSites', 'cys'))
        and row['sequence']
        and not set(row['sequence']) - AA
    ):
        return 'unmodified'
    if (
        row['modifications']
        or row['variableSites']
        or row['cys']
        or row['nterm'] not in ('', 'Hydrogen')
        or row['cterm'] not in ('', 'Free Acid')
    ):
        return 'modified'
    return 'unknown'


def restriction(filename):
    match = re.fullmatch(r'(D[PRQ]B)(\d{4})_(D[PRQ]A)(\d{4})_(.+)\.PSMexport\.\d+\.ssv', filename)
    if match is None:
        raise ValueError('Unreviewed source filename ' + filename)
    beta, b, alpha, a, _ = match.groups()
    # File nomenclature uses DRB for DRB1 in this source; other loci are A1/B1.
    return (
        f'HLA-{alpha}1*{a[:2]}:{a[2:]}/{beta}1*{b[:2]}:{b[2:]}'
        if alpha != 'DRA'
        else f'HLA-DRA*{a[:2]}:{a[2:]}/DRB1*{b[:2]}:{b[2:]}'
    )


def recover(accepted, sources, root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    with accepted.open() as stream:
        allowed = {(r['peptide'], r['mhc_restriction']) for r in csv.DictReader(stream)}
    forms = defaultdict(list)
    counts = Counter()
    source_rows = []
    for spec in sources:
        path = Path(spec['path'])
        assert sha(path) == spec['sha256']
        mhc = restriction(path.name)
        with path.open() as stream:
            for ordinal, row in enumerate(csv.DictReader(stream, delimiter=';'), 2):
                state = chemistry(row)
                counts['psm:' + state] += 1
                peptide = row['sequence'].upper()
                if (peptide, mhc) not in allowed:
                    counts['outside_published_ligand_set'] += 1
                    continue
                # Do not use search hits to known spike-ins/contaminants as evidence.
                if not any(
                    x == 'Human' or x.startswith(('nuORFdb_human', 'TCGA-'))
                    for x in row['species'].split('|')
                ):
                    counts['nonhuman_or_control'] += 1
                    continue
                identity = (peptide, mhc, state)
                forms[identity].append(
                    dict(
                        source_file=path.name,
                        source_sha256=spec['sha256'],
                        source_row=ordinal,
                        source_psm_number=row['number'],
                        source_filename=row['filename'],
                        **{
                            k: row[k]
                            for k in ('modifications', 'variableSites', 'nterm', 'cterm', 'cys')
                        },
                    )
                )
        source_rows.append(dict(path=str(path), sha256=spec['sha256']))
    path = root / 'strazar_2023_recovered_unmodified.csv'
    with (
        path.open('w', newline='') as stream,
        (root / 'source-evidence.jsonl').open('w') as evidence,
    ):
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                'peptide',
                'mhc_class',
                'mhc_restriction',
                'has_ptm',
                'peptide_modifications',
                'source_evidence_id',
            ],
        )
        writer.writeheader()
        for (peptide, mhc, state), support in sorted(forms.items()):
            counts['published_pmhc:' + state] += 1
            key = hashlib.sha256(json.dumps([peptide, mhc, state]).encode()).hexdigest()
            evidence.write(
                json.dumps(
                    dict(
                        evidence_id=key,
                        peptide=peptide,
                        mhc_restriction=mhc,
                        state=state,
                        support=support,
                    )
                )
                + '\n'
            )
            if state == 'unmodified':
                writer.writerow(
                    dict(
                        peptide=peptide,
                        mhc_class='II',
                        mhc_restriction=mhc,
                        has_ptm=False,
                        peptide_modifications='Unmodified',
                        source_evidence_id=key,
                    )
                )
    summary = dict(
        status='partial_source_recovery',
        accepted_ligand_set_sha256=sha(accepted),
        counts=dict(counts),
        sources=source_rows,
        output=dict(path=str(path), sha256=sha(path)),
        evidence=dict(
            path=str(root / 'source-evidence.jsonl'), sha256=sha(root / 'source-evidence.jsonl')
        ),
        policy=(
            'exact published ligand/MHC intersection; explicit native terminal groups + '
            'no variable/fixed modifications; human source support; '
            'preserve co-observed modified forms separately'
        ),
        seconds=time.monotonic() - started,
        limitations=(
            'Only successfully downloaded PSM files reviewed; no negative chemistry claim '
            'for remaining ligand rows. No model/test outcomes consulted.'
        ),
    )
    return summary


def main():
    accepted = Path('/Users/iskander/Library/Caches/hitlist/strazar_2023_hla2.csv')
    receipts = [
        F / 'results/acquisition-strazar.json',
        F / 'results/acquisition-strazar-resume.json',
    ]
    sources = list(
        {
            r['id']: r
            for receipt in receipts
            if receipt.exists()
            for r in json.loads(receipt.read_text())
            if r['status'] == 'downloaded'
        }.values()
    )
    summary = recover(accepted, sources, ROOT)
    (F / 'results/strazar-recovery.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(counts=summary['counts'], seconds=summary['seconds'])))


if __name__ == '__main__':
    main()
