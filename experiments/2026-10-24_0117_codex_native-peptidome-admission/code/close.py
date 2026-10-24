"""Verify audit evidence and freeze its publication inventory; no training claim."""
import hashlib
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((FAMILY / relative).read_text())


def main():
    for phase in ['audit', 'canonical_audit', 'qualify']:
        receipt = read('reproduce/' + phase + '-execution.json')
        assert receipt['returncode'] == 0
        assert digest(FAMILY / 'code' / (phase + '.py')) == receipt['source_sha256']
        assert digest(FAMILY / 'reproduce/source' / (phase + '.py')) == receipt['source_sha256']
    audit = read('results/workbook-audit.json')
    canonical = read('results/canonical-audit.json')
    semantic = read('results/semantic-verification.json')
    assert audit['n_parser_disagreements'] == 0
    assert canonical['source_value_replay_pass'] and not canonical['training_admission']
    assert canonical['n_ratio_slots_replayed'] == canonical['n_query_rows'] == semantic['n_quantitative_parses_verified']
    assert sum(semantic['value_status'].values()) == canonical['n_ratio_slots_replayed']
    for asset in read('results/retrieval.json')['assets']:
        assert digest(Path(asset['path'])) == asset['sha256']
    package = Path(canonical['package_path'])
    assert digest(package / 'manifest.json') == canonical['package_manifest_sha256']
    assert read('results/canonical-package-manifest.json') == json.loads((package / 'manifest.json').read_text())
    for relative, receipt in read('results/canonical-package-manifest.json')['files'].items():
        assert digest(package / relative) == receipt['sha256']
    issue = read('results/hitlist-791-readback.json')
    assert issue['body'] == (FAMILY / 'results/hitlist-bug-refined.md').read_text()
    assert issue['state'] == 'open'
    conclusion = dict(
        all_pass=True, scope='Available-source audit closure only', training_admitted=False,
        biological_model_trained=False, full_goal_complete=False,
        verified_phases=['workbook source preservation', 'canonical row/query replay', 'quantitative parsing'],
        semantic_admission='failed; Hitlist791 corrections and source adjudication required',
        unresolved_sources=['Original2012 epoxomicin table; source requests failed'],
    )
    (FAMILY / 'results/closure.json').write_text(json.dumps(conclusion, indent=2) + '\n')
    files = {str(p.relative_to(FAMILY)): digest(p) for p in sorted(FAMILY.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'closure-hashes.json'}
    (FAMILY / 'reproduce/closure-hashes.json').write_text(json.dumps(dict(files=files), indent=2) + '\n')
    print(json.dumps(dict(**conclusion, n_frozen_files=len(files)), indent=2))


if __name__ == '__main__':
    main()
