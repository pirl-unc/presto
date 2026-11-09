"""Live side-by-side results and receipt-verified automatic experiment closure."""

import functools
import gzip
import hashlib
import http.server
import json
from pathlib import Path
import shutil
import tarfile
import threading
import time

import modal
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]
CONDITIONS = [
    f'{mode}-s{seed}' for mode in ('pooled', 'calibration', 'factorized') for seed in (42, 43)
]
STORE = modal.Dict.from_name('presto-cross-class-live')
VOLUME = modal.Volume.from_name('presto-checkpoints')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collect(condition, receipt):
    destination = ROOT / 'artifacts' / F.name / 'collected'
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (condition + '.tar.gz')
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        with temporary.open('wb') as stream:
            for block in VOLUME.read_file(receipt['archive']):
                stream.write(block)
        assert temporary.stat().st_size == receipt['bytes'] and sha(temporary) == receipt['sha256']
        temporary.replace(archive)
    assert sha(archive) == receipt['sha256']
    run = destination / condition
    if not (run / 'report.json').exists():
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                assert parts and parts[0] == condition and '..' not in parts
                assert not member.name.startswith('/') and (member.isfile() or member.isdir())
            tar.extractall(destination, filter='data')
    report = json.loads((run / 'report.json').read_text())
    assert sha(run / 'raw/selected.pt') == report['checkpoint_sha256']
    tables, memberships, verified = [], {}, []
    for split in ('validation', 'test'):
        with gzip.open(run / (split + '-predictions.jsonl.gz'), 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        # Recompute outcomes from serialized observations, independently of model losses.
        for row in rows:
            observed = row['observation']
            if row['endpoint'] == 'ic50':
                assert np.isclose(row['target'], np.log10(observed['value']), atol=1e-12)
                method = row['assay_method']
                key = '/'.join(
                    ['ic50', method['preparation'], method['mechanism'], method['detector']]
                )
                if key in row['assay_panel'] and method['review_status'] in (
                    'verified',
                    'source_tag_unreviewed',
                ):
                    assert np.isclose(row['prediction'], row['assay_panel'][key], atol=1e-6)
        memberships[split] = hashlib.sha256(
            json.dumps(
                sorted(
                    (r['observation_id'], r['target'], r['qualifier'], r['observation'])
                    for r in rows
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for cls in ('I', 'II'):
            for detector in ('radioactivity', 'fluorescence'):
                exact = [
                    r
                    for r in rows
                    if r['endpoint'] == 'ic50'
                    and r['qualifier'] == '='
                    and r['mhc_classes'] == [cls]
                    and r['assay_method']['detector'] == detector
                    and r['assay_method']['review_status'] == 'source_tag_unreviewed'
                ]
                if not exact:
                    continue
                truth, pred = (
                    np.array([r['target'] for r in exact]),
                    np.array([r['prediction'] for r in exact]),
                )
                rho = (
                    float(spearmanr(truth, pred).statistic)
                    if np.ptp(truth) and np.ptp(pred)
                    else None
                )
                rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
                reported = report['summary']['metrics'][split]['ic50']['by_assay_method'][
                    'purified/competitive/' + detector + '/source_tag_unreviewed'
                ]['by_class'][cls]
                assert reported['n'] == len(exact) and np.isclose(reported['rmse'], rmse, atol=1e-8)
                assert (
                    reported['spearman'] == rho
                    if rho is None
                    else np.isclose(reported['spearman'], rho, atol=1e-8)
                )
                tables.append(
                    dict(
                        condition=condition,
                        split=split,
                        mhc_class=cls,
                        detector=detector,
                        n=len(exact),
                        spearman=rho,
                        rmse=rmse,
                    )
                )
        for kind, endpoints in report['ligands'][split].items():
            negative = [
                r
                for r in rows
                if r['endpoint'] == 'elution_observation'
                and r['observation']['provenance'].get('background_kind') == kind
            ]
            parents = {r['observation']['provenance']['parent_observation_id'] for r in negative}
            population = negative + [
                r
                for r in rows
                if r['endpoint'] == 'elution_observation' and r['observation_id'] in parents
            ]
            y, p = [r['target'] for r in population], [r['prediction'] for r in population]
            if len(set(y)) == 2:
                assert np.isclose(
                    roc_auc_score(y, p), endpoints['elution_observation']['auroc'], atol=1e-8
                )
                assert np.isclose(
                    average_precision_score(y, p),
                    endpoints['elution_observation']['auprc'],
                    atol=1e-8,
                )
                verified.append(split + ':' + kind)
    review = F / 'results' / condition
    review.mkdir(exist_ok=True)
    for name in ('report.json', 'hardware.json'):
        shutil.copyfile(run / name, review / name)
    for name in (
        'summary.json',
        'history.json',
        'coverage.json',
        'admission.json',
        'resolved-config.json',
    ):
        shutil.copyfile(run / 'raw' / name, review / name)
    result = dict(
        status='independently_verified',
        directory=str(run),
        receipt=receipt,
        checkpoint_sha256=report['checkpoint_sha256'],
        memberships=memberships,
        ic50=tables,
        ligand_checks=verified,
        predictions={
            split: {
                'path': str(run / (split + '-predictions.jsonl.gz')),
                'sha256': sha(run / (split + '-predictions.jsonl.gz')),
            }
            for split in ('validation', 'test')
        },
    )
    write(review / 'collection.json', result)
    return result


def close(results):
    for split in ('validation', 'test'):
        assert len({r['memberships'][split] for r in results.values()}) == 1, (
            'comparison populations differ'
        )
    cells = [cell for r in results.values() for cell in r['ic50']]
    write(
        F / 'results/comparison.json',
        dict(
            status='verified',
            ic50=cells,
            interpretation='retrospective development comparison; no automatic model promotion',
        ),
    )
    lines = [
        '\n<!-- verified-assay-closeout -->',
        '\n## Verified matched results\n',
        '| Condition | Split | Class | Detector | Exact n | Spearman | RMSE log10(nM) |',
        '|---|---|---|---|---:|---:|---:|',
    ]
    for cell in cells:
        rho = 'unavailable' if cell['spearman'] is None else f"{cell['spearman']:.4f}"
        lines.append(
            f"| {cell['condition']} | {cell['split']} | {cell['mhc_class']} | "
            f"{cell['detector']} | {cell['n']} | {rho} | {cell['rmse']:.4f} |"
        )
    lines += [
        '',
        'All six validation/test populations and serialized observations match.',
        'Scipy independently reproduces IC50 correlations; scikit-learn independently',
        'reproduces available ligand-background AUROC/AUPRC. Full endpoint/study and',
        'ligand metrics, parameter coverage and actual hardware are in each results/',
        'condition report; collection.json records every prediction dump and hash.',
        'Winner/promotion remains pending scientific review and study-aware uncertainty.',
    ]
    text = '\n'.join(lines) + '\n'
    readme = F / 'README.md'
    if '<!-- verified-assay-closeout -->' not in readme.read_text():
        with readme.open('a') as stream:
            stream.write(text)
    log = F.parent / 'experiment_log.md'
    marker = '<!-- closed-' + F.name + ' -->'
    if marker not in log.read_text():
        with log.open('a') as stream:
            stream.write(
                '\n'
                + marker
                + '\n## 2026-10-29 — Assay measurement comparison: verified results\n\n'
            )
            stream.write('Codex / GPT-6. [Full contract and results](' + F.name + '/README.md).\n')
            stream.write(text)
    write(
        F / 'results/closure.json',
        dict(status='metrics_verified_review_pending', conditions=CONDITIONS),
    )


def collector():
    results = {}
    while True:
        try:
            for name in CONDITIONS:
                if name in results:
                    continue
                value = STORE.get(F.name + ':' + name, None)
                if value and value.get('status') == 'finished':
                    results[name] = collect(name, value['result'])
                    print('Verified', name, flush=True)
            if len(results) == len(CONDITIONS):
                close(results)
                return
        except Exception as exc:
            write(
                F / 'results/collection-error.json', dict(error=repr(exc), updated_unix=time.time())
            )
            print('Collector retry:', repr(exc), flush=True)
        time.sleep(30)


def main():
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 54504),
        functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(F)),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=collector, daemon=True).start()
    while True:
        try:
            states = {
                name: STORE.get(F.name + ':' + name, None) for name in ['coordinator', *CONDITIONS]
            }
            coordinator = states.pop('coordinator') or {}
            if 'preparation' in coordinator:
                write(F / 'results/prepared.json', coordinator['preparation'])
            # Receipt reports are collected separately; keep live payload compact.
            runs = [
                {
                    k: v
                    for k, v in (states[name] or dict(condition=name, status='pending')).items()
                    if k != 'result'
                }
                for name in CONDITIONS
            ]
            write(
                F / 'results/live.json',
                dict(
                    refreshed_unix=time.time(),
                    runs=runs,
                    coordinator={k: v for k, v in coordinator.items() if k != 'preparation'},
                ),
            )
        except Exception as exc:
            print('Poll retry:', repr(exc), flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
