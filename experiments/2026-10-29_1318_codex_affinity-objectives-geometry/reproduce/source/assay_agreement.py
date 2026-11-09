"""TRAIN-only concentration agreement; one vote per aggregated pMHC context."""

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

import numpy as np
from scipy.stats import pearsonr, spearmanr

ENDPOINTS = ('ic50', 'kd', 'ec50', 'kd_via_ic50', 'kd_via_ec50')
NM = {'pM': 0.001, 'nM': 1.0, 'uM': 1000.0, 'mM': 1e6, 'M': 1e9}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scores(rows):
    x = np.asarray([r['a_log10_nM'] for r in rows], dtype=float)
    y = np.asarray([r['b_log10_nM'] for r in rows], dtype=float)
    varied = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
    difference = y - x
    return dict(
        n=len(rows),
        unique_pmhcs=len({r['pmhc_sha256'] for r in rows}),
        study_components=len({s for r in rows for s in r['studies_a'] + r['studies_b']}),
        spearman=float(spearmanr(x, y).statistic) if varied else None,
        pearson=float(pearsonr(x, y).statistic) if varied else None,
        difference_rmse_log10_nM=float(np.sqrt(np.mean(difference**2))) if len(x) else None,
        median_b_minus_a_log10_nM=float(np.median(difference)) if len(x) else None,
        identical_aggregate_fraction=float(np.mean(x == y)) if len(x) else None,
        interpretation=(
            'measurement agreement, not prediction accuracy; shared contexts may be dependent'
        ),
    )


def table(path, rows):
    columns = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )


def audit(sources, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    exact, all_pmhcs, methods, counts, panel_fields = (
        defaultdict(list),
        defaultdict(set),
        Counter(),
        Counter(),
        Counter(),
    )
    source_records, seen, original_endpoints = [], set(), defaultdict(set)
    for path, expected in sources:
        path = Path(path)
        h = hashlib.sha256()
        n = 0
        with path.open('rb') as stream:
            for line in stream:
                h.update(line)
                row = json.loads(line)
                n += 1
                assert row['split'] == 'train'
                assert row['observation_id'] not in seen
                seen.add(row['observation_id'])
                counts['source_rows'] += 1
                endpoint = row['endpoint']
                if endpoint not in ENDPOINTS:
                    counts['outside_concentration_outputs:' + endpoint] += 1
                    continue
                assert len(row['mhc']) == 1
                mhc = row['mhc'][0]
                cls = mhc.get('class_override') or mhc['mhc_class']
                assert cls in ('I', 'II')
                pmhc = digest((row['peptide'], mhc['groove1'], mhc['groove2'], cls))
                prov = row['provenance']
                raw = prov['raw_record']
                group = row['group_id']
                assert group
                counts[f'concentration:{endpoint}:{cls}:{row["qualifier"]}'] += 1
                all_pmhcs[cls, endpoint].add(pmhc)
                methods[
                    endpoint,
                    cls,
                    raw.get('assay_method', ''),
                    raw.get('response_measured', ''),
                    row['unit'],
                    row['qualifier'],
                ] += 1
                original_endpoints[
                    prov['measurement_namespace'], prov['original_measurement_id']
                ].add(endpoint)
                for field in (
                    'assay_comments',
                    'attributed_sample_label',
                    'culture_condition',
                    'supplementary_file',
                ):
                    panel_fields[endpoint, field, bool(raw.get(field))] += 1
                if row['qualifier'] != '=':
                    continue
                value = row['value'] * NM[row['unit']]
                assert math.isfinite(value) and value > 0
                exact[cls, pmhc, endpoint, group].append(math.log10(value))
        assert h.hexdigest() == expected, path
        source_records.append(
            dict(path=str(path), sha256=expected, rows=n, bytes=path.stat().st_size)
        )
    conflicts = sum(len(v) > 1 for v in original_endpoints.values())
    assert conflicts == 0, 'same original assay identity occurs under multiple outputs'
    medians = {key: statistics.median(values) for key, values in exact.items()}
    by_pmhc = defaultdict(dict)
    for (cls, pmhc, endpoint, study), value in medians.items():
        by_pmhc[cls, pmhc, endpoint][study] = value
    comparisons, paired_values = [], []
    for cls in ('I', 'II'):
        for a, b in combinations(ENDPOINTS, 2):
            overlaps = sorted(all_pmhcs[cls, a] & all_pmhcs[cls, b])
            populations = defaultdict(list)
            for pmhc in overlaps:
                left = by_pmhc.get((cls, pmhc, a), {})
                right = by_pmhc.get((cls, pmhc, b), {})
                if not left or not right:
                    continue
                common = set(left) & set(right)
                base = dict(mhc_class=cls, endpoint_a=a, endpoint_b=b, pmhc_sha256=pmhc)
                for study in sorted(common):
                    populations['within_study'].append(
                        dict(
                            **base,
                            scope='within_study',
                            a_log10_nM=left[study],
                            b_log10_nM=right[study],
                            studies_a=[study],
                            studies_b=[study],
                        )
                    )
                pooled = dict(
                    **base,
                    scope='pooled_pmhc',
                    a_log10_nM=statistics.median(left.values()),
                    b_log10_nM=statistics.median(right.values()),
                    studies_a=sorted(left),
                    studies_b=sorted(right),
                )
                populations['pooled_pmhc'].append(pooled)
                if not common:
                    populations['disjoint_studies'].append({**pooled, 'scope': 'disjoint_studies'})
            for scope in ('within_study', 'pooled_pmhc', 'disjoint_studies'):
                rows = populations[scope]
                comparisons.append(
                    dict(
                        mhc_class=cls,
                        endpoint_a=a,
                        endpoint_b=b,
                        scope=scope,
                        overlap_including_bounds=len(overlaps),
                        **scores(rows),
                    )
                )
                paired_values.extend(rows)
    repeat_rows = []
    for (cls, pmhc, endpoint), studies in by_pmhc.items():
        values = [v for study in studies for v in exact[cls, pmhc, endpoint, study]]
        if len(values) < 2:
            continue
        repeat_rows.append(
            dict(
                mhc_class=cls,
                pmhc_sha256=pmhc,
                endpoint=endpoint,
                n=len(values),
                study_components=len(studies),
                range_log10_nM=max(values) - min(values),
                iqr_log10_nM=float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            )
        )
    repeat_summary = []
    for cls in ('I', 'II'):
        for endpoint in ENDPOINTS:
            rows = [r for r in repeat_rows if r['mhc_class'] == cls and r['endpoint'] == endpoint]
            repeat_summary.append(
                dict(
                    mhc_class=cls,
                    endpoint=endpoint,
                    n_repeated_pmhcs=len(rows),
                    n_with_multiple_studies=sum(r['study_components'] > 1 for r in rows),
                    median_range_log10_nM=statistics.median(r['range_log10_nM'] for r in rows)
                    if rows
                    else None,
                    interpretation=(
                        'repeated pMHC observations; not necessarily technical replicates'
                    ),
                )
            )
    method_rows = [
        dict(endpoint=e, mhc_class=c, assay_method=m, response_measured=r, unit=u, qualifier=q, n=n)
        for (e, c, m, r, u, q), n in sorted(methods.items())
    ]
    with gzip.open(output / 'paired-values.jsonl.gz', 'wt') as stream:
        for row in paired_values:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    table(output / 'correlations.csv', comparisons)
    table(output / 'assay-methods.csv', method_rows)
    table(output / 'repeat-spread.csv', repeat_rows)
    result = dict(
        status='completed_source_audit',
        sources=source_records,
        counts=dict(counts),
        exact_observation_count=sum(map(len, exact.values())),
        aggregated_pmhc_endpoint_study_count=len(exact),
        comparisons=comparisons,
        assay_methods=method_rows,
        repeat_summary=repeat_summary,
        panel_metadata_availability=[
            dict(endpoint=e, field=f, present=p, n=n)
            for (e, f, p), n in sorted(panel_fields.items())
        ],
        original_assay_cross_endpoint_conflicts=conflicts,
        seconds=time.monotonic() - started,
        validation_test_used=False,
        model_fitted=False,
        synthetic_data='none; original quantitative TRAIN observations only',
    )
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
