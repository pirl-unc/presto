"""Exercise document closure using historical parent artifacts in a temporary tree."""

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time

import compare_parent as analyzer

F = Path(__file__).resolve().parents[1]
P = F.parent / '2026-10-28_1540_codex_temporal-multi-output'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    with tempfile.TemporaryDirectory(prefix='presto-closeout-check-') as name:
        target = Path(name) / F.name
        target.mkdir()
        readme = (F / 'README.md').read_text()
        (target / 'README.md').write_text(readme)
        before, after = (
            '# Registry\n\nEarlier entry unchanged.\n\n',
            '\n## Later entry\n\nUnchanged.\n',
        )
        source_registry = (F.parent / 'experiment_log.md').read_text()
        entry = source_registry.split('## ' + F.name, 1)[1].split('\n## ', 1)[0]
        registry = before + '## ' + F.name + entry + after
        (target.parent / 'experiment_log.md').write_text(registry)
        analyzer.F = target
        gates = {
            'reporting-receipt': 'completed',
            'prediction-verification': 'passed',
            'cohort-composition-verification': 'passed',
            'parent-comparison': 'passed',
        }
        for key, status in gates.items():
            write(target / 'results' / (key + '.json'), {'status': status})
        gate = target / 'results/prediction-verification.json'
        write(gate, {'status': 'pending'})
        try:
            analyzer.closeout()
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete verification did not block publication')
        assert (target / 'README.md').read_text() == readme
        assert (target.parent / 'experiment_log.md').read_text() == registry
        assert not (target / 'results/closeout.json').exists()
        write(gate, {'status': 'passed'})
        for item in ('validation-decision.json', 'stratified-metrics.json'):
            # Historical real artifacts exercise the schema; these are not new results.
            shutil.copyfile(P / 'results' / item, target / 'results' / item)
        selection = json.loads((P / 'results/validation-decision.json').read_text())
        for condition in selection['scores']:
            write(
                target / 'results' / condition / 'report.json',
                json.loads((P / 'results' / condition / 'report.json').read_text()),
            )
        analyzer.closeout()
        first_readme = (target / 'README.md').read_text()
        first_registry = (target.parent / 'experiment_log.md').read_text()
        assert 'both fits completed' in first_readme
        assert 'Runtime/peak memory remain pending' not in first_readme
        assert '| None |' not in first_readme
        assert 'No new predictive result yet' not in first_readme
        assert first_registry.startswith(before) and first_registry.endswith(after)
        analyzer.closeout()
        assert (target / 'README.md').read_text() == first_readme
        assert (target.parent / 'experiment_log.md').read_text() == first_registry
    analyzer.F = F
    snapshot = F / 'reproduce/analysis-before-predictions-v2'
    snapshot.mkdir(exist_ok=False)
    for source in (Path(__file__), Path(analyzer.__file__)):
        shutil.copyfile(source, snapshot / source.name)
    result = dict(
        status='passed',
        verified_unix=time.time(),
        gates_fail_closed=True,
        actual_parent_result_schema_compatible=True,
        unrelated_registry_entries_preserved=True,
        repeated_closeout_idempotent=True,
        binary_counts_not_null=True,
        stale_pending_wording_removed=True,
        scope='Temporary historical-result replay; active runs are not marked complete',
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()},
    )
    write(F / 'results/closeout-readiness-v2.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
