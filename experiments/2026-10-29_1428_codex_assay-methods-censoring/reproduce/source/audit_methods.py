"""Descriptive assay-method and possible censoring audit; never changes qualifiers."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import statistics
import time

from scipy.stats import pearsonr, spearmanr

ENDPOINTS = ('ic50', 'kd', 'ec50', 'kd_via_ic50', 'kd_via_ec50')
THRESHOLDS = (5000, 10000, 20000, 25000, 50000, 100000)
NM = {'pM': 0.001, 'nM': 1.0, 'uM': 1000.0, 'mM': 1e6, 'M': 1e9}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def method_axes(method):
    parts = (method or '').split('/')
    return dict(
        preparation=parts[0] or 'unknown',
        mechanism=parts[1] if len(parts) > 1 else 'unknown',
        readout='/'.join(parts[2:]) if len(parts) > 2 else 'unknown',
    )


def csv_write(path, rows):
    keys = sorted({k for row in rows for k in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            )


def heap_candidates(records):
    """Screen a single study/MHC/method/endpoint group, leaving all records untouched."""
    if len(records) < 20:
        return []
    by_value = defaultdict(list)
    for r in records:
        by_value[r['value_nM']].append(r)
    result = []
    for value, rows in sorted(by_value.items()):
        exact = [r for r in rows if r['qualifier'] == '=']
        if len(exact) < 5 or len(exact) / len(records) < 0.1:
            continue
        negative = sum(r['qualitative'].lower().startswith('negative') for r in exact)
        right = sum(r['qualifier'] in ('>', '>=') for r in rows)
        lower = sum(r['qualifier'] in ('<', '<=') for r in rows)
        above = sum(r['value_nM'] > value for r in records)
        below = sum(r['value_nM'] < value for r in records)
        result.append(
            dict(
                value_nM=value,
                group_rows=len(records),
                exact_at_value=len(exact),
                fraction_of_group=len(exact) / len(records),
                qualitative_negative_exact=negative,
                explicit_right_bounds_at_value=right,
                explicit_left_bounds_at_value=lower,
                reported_values_above=above,
                reported_values_below=below,
                exact_missing_raw_qualifier=sum(r['raw_qualifier'] in ('', None) for r in exact),
                right_limit_review_candidate=bool(
                    right or (above == 0 and negative / len(exact) >= 0.5)
                ),
                left_limit_review_candidate=bool(lower),
                assay_ids=[r['assay_id'] for r in exact],
                interpretation='Screening evidence only; original qualifiers remain unchanged.',
            )
        )
    return result


def correlations(records):
    pairs, metrics = [], []
    methods = sorted({r['method'] for r in records})
    for cls in ('I', 'II'):
        for response in ('all_concentration_labels',) + ENDPOINTS:
            selected = [
                r
                for r in records
                if r['mhc_class'] == cls
                and r['qualifier'] == '='
                and (response == 'all_concentration_labels' or r['endpoint'] == response)
            ]
            grouped = defaultdict(list)
            for r in selected:
                grouped[r['pmhc_sha256'], r['method'], r['group_id']].append(
                    math.log10(r['value_nM'])
                )
            medians = defaultdict(dict)
            for (pid, method, study), values in grouped.items():
                medians[pid, method][study] = statistics.median(values)
            for a, b in combinations(methods, 2):
                left = {pid for pid, method in medians if method == a}
                right = {pid for pid, method in medians if method == b}
                values = []
                for pid in sorted(left & right):
                    x, y = medians[pid, a], medians[pid, b]
                    values.append(
                        dict(
                            mhc_class=cls,
                            response_stratum=response,
                            method_a=a,
                            method_b=b,
                            pmhc_sha256=pid,
                            a_log10_nM=statistics.median(x.values()),
                            b_log10_nM=statistics.median(y.values()),
                            studies_a=sorted(x),
                            studies_b=sorted(y),
                            common_studies=sorted(set(x) & set(y)),
                        )
                    )
                x = [r['a_log10_nM'] for r in values]
                y = [r['b_log10_nM'] for r in values]
                varied = len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1
                metrics.append(
                    dict(
                        mhc_class=cls,
                        response_stratum=response,
                        method_a=a,
                        method_b=b,
                        n=len(values),
                        spearman=float(spearmanr(x, y).statistic) if varied else None,
                        pearson=float(pearsonr(x, y).statistic) if varied else None,
                        rmse_difference_log10_nM=math.sqrt(
                            sum((v - u) ** 2 for u, v in zip(x, y)) / len(x)
                        )
                        if x
                        else None,
                        median_b_minus_a=statistics.median(v - u for u, v in zip(x, y))
                        if x
                        else None,
                        n_with_any_shared_study=sum(bool(r['common_studies']) for r in values),
                    )
                )
                pairs.extend(values)
    return metrics, pairs


def train_audit(source, expected, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, study_metadata, counts, method_counts, thresholds = (
        [],
        {},
        Counter(),
        Counter(),
        Counter(),
    )
    h = hashlib.sha256()
    with Path(source).open('rb') as stream:
        for line in stream:
            h.update(line)
            r = json.loads(line)
            assert r['split'] == 'train'
            counts['physical_rows'] += 1
            if r['endpoint'] not in ENDPOINTS:
                continue
            mhc = r['mhc'][0]
            assert len(r['mhc']) == 1
            cls = mhc.get('class_override') or mhc['mhc_class']
            p, raw = r['provenance'], r['provenance']['raw_record']
            method = raw['assay_method']
            value = r['value'] * NM[r['unit']]
            record = dict(
                pmhc_sha256=digest((r['peptide'], mhc['groove1'], mhc['groove2'], cls)),
                mhc_sha256=digest((mhc['groove1'], mhc['groove2'], cls)),
                mhc_class=cls,
                endpoint=r['endpoint'],
                method=method,
                group_id=r['group_id'],
                value_nM=value,
                qualifier=r['qualifier'],
                raw_qualifier=raw.get('measurement_inequality'),
                qualitative=raw.get('qualitative_measurement') or '',
                assay_id=p['original_measurement_id'],
                source=r['source'],
            )
            rows.append(record)
            counts['concentration_rows'] += 1
            counts['qualifier:' + r['qualifier']] += 1
            qkind = 'missing' if record['raw_qualifier'] in ('', None) else record['raw_qualifier']
            counts['raw_qualifier:' + qkind] += 1
            method_counts[cls, r['endpoint'], method, r['qualifier']] += 1
            if value in THRESHOLDS:
                thresholds[cls, r['endpoint'], method, value, r['qualifier'], qkind] += 1
            meta = dict(
                study_id=p['study_id'],
                group_aliases=p['group_aliases'],
                **{
                    k: raw.get(k)
                    for k in ('reference_iri', 'reference_title', 'pmid', 'submission_id')
                },
            )
            key = r['group_id']
            if key not in study_metadata:
                study_metadata[key] = dict(metadata=meta, comments=Counter())
            if raw.get('assay_comments'):
                study_metadata[key]['comments'][raw['assay_comments']] += 1
    assert h.hexdigest() == expected
    assert len({r['assay_id'] for r in rows}) == len(rows)
    with gzip.open(output / 'concentration-records.jsonl.gz', 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    groups = defaultdict(list)
    for r in rows:
        groups[r['mhc_class'], r['mhc_sha256'], r['group_id'], r['method'], r['endpoint']].append(r)
    heaps = []
    for (cls, mhc, group, method, endpoint), values in groups.items():
        heaps.extend(
            dict(
                mhc_class=cls, mhc_sha256=mhc, group_id=group, method=method, endpoint=endpoint, **r
            )
            for r in heap_candidates(values)
        )
    metrics, pairs = correlations(rows)
    with gzip.open(output / 'method-pairs.jsonl.gz', 'wt') as stream:
        for row in pairs:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    method_rows = [
        dict(mhc_class=c, endpoint=e, method=m, qualifier=q, n=n, **method_axes(m))
        for (c, e, m, q), n in sorted(method_counts.items())
    ]
    threshold_rows = [
        dict(mhc_class=c, endpoint=e, method=m, value_nM=v, qualifier=q, raw_qualifier=k, n=n)
        for (c, e, m, v, q, k), n in sorted(thresholds.items())
    ]
    for name, data in [
        ('methods', method_rows),
        ('threshold-values', threshold_rows),
        ('heap-candidates', heaps),
        ('method-correlations', metrics),
    ]:
        csv_write(output / (name + '.csv'), data)
    (output / 'heap-candidates.json').write_text(json.dumps(heaps, indent=2) + '\n')
    (output / 'study-metadata.json').write_text(json.dumps(study_metadata, indent=2) + '\n')
    summary = dict(
        status='completed_train_audit',
        counts=dict(counts),
        methods=method_rows,
        method_correlations=metrics,
        threshold_values=threshold_rows,
        heap_candidate_groups=len(heaps),
        heaped_exact_rows=sum(r['exact_at_value'] for r in heaps),
        right_limit_candidate_exact_rows=sum(
            r['exact_at_value'] for r in heaps if r['right_limit_review_candidate']
        ),
        raw_source=dict(path=str(source), sha256=expected),
        seconds=time.monotonic() - started,
        changed_qualifiers=0,
        validation_test_used=False,
        model_fitted=False,
    )
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def source_census(specs, output):
    import pyarrow.parquet as pq

    counts = Counter()
    for name, spec in specs.items():
        path = Path(spec['path'])
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == spec['sha256']
        fields = [
            'source',
            'mhc_class',
            'mhc_species',
            'assay_method',
            'response_measured',
            'measurement_units',
            'mhc_restriction',
            'has_ptm',
            'allele_resolution',
        ]
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_rows == spec['rows']
        for batch in pf.iter_batches(columns=fields, batch_size=65536):
            for r in batch.to_pylist():
                restriction = r.pop('mhc_restriction') or ''
                r['explicit_hla_name'] = restriction.startswith('HLA-')
                counts[name, json.dumps(r, sort_keys=True)] += 1
    census = [dict(index=n, **json.loads(key), n=count) for (n, key), count in counts.items()]
    csv_write(Path(output) / 'source-census.csv', census)
    (Path(output) / 'source-census.json').write_text(json.dumps(census, indent=2) + '\n')
    return dict(rows=sum(counts.values()), strata=len(counts), sources=specs)
