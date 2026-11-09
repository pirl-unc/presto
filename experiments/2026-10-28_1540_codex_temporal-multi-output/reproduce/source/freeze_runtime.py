"""Freeze the complete installable package and exact sequence/proteome references."""

from pathlib import Path
import hashlib, json, shutil, subprocess, tomllib

f = Path(__file__).resolve().parents[1]
root = f.parents[1]
upload = root / 'artifacts' / f.name / 'upload'
upload.mkdir(parents=True, exist_ok=True)
if (upload / 'runtime-pins.json').exists():
    raise SystemExit('Runtime already frozen')
config = tomllib.loads((root / 'pyproject.toml').read_text())
directories = {
    Path(p.removeprefix('presto').lstrip('.').replace('.', '/'))
    for p in config['tool']['setuptools']['packages']
}
files = sorted({str(p.relative_to(root)) for d in directories for p in (root / d).glob('*.py')})
files.append('data/b2m_sequences.csv')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


for name in files:
    destination = upload / 'presto' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, destination)
catalog = Path('/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv')
proteome = Path(
    '/Users/iskander/Library/Caches/pyensembl/GRCh38/ensembl112/7fa50041eebe6c62f62f5f3f009bb956e10e294d7eb748bae30f40ffc7ecf070-Homo_sapiens.GRCh38.pep.all.fa.gz'
)
shutil.copyfile(catalog, upload / 'mhc-full-seqs.csv')
shutil.copyfile(proteome, upload / 'human-ensembl112.pep.fa.gz')
pins = {
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_dirty': subprocess.check_output(['git', 'status', '--short'], text=True),
    'files': {p: sha(upload / 'presto' / p) for p in files},
    'catalog': {'path': str(catalog), 'sha256': sha(catalog)},
    'proteome': {
        'path': str(proteome),
        'sha256': sha(proteome),
        'source': 'Ensembl release 112 human GRCh38 all peptide FASTA',
        'url': 'https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz',
    },
    'pretraining': False,
    'shared_test_result': '2797 passed, 4 skipped, 33 warnings in 350.52s; subsequent literal purified-method fix: 79 focused tests passed in 12.38s',
}
for p in (upload / 'runtime-pins.json', f / 'reproduce/runtime-pins.json'):
    p.write_text(json.dumps(pins, indent=2) + '\n')
print('froze', len(files), 'package files at', pins['git_commit'])
