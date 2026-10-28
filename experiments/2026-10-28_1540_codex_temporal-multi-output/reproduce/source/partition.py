"""Frozen metadata-only paper selection; no prediction or target magnitudes used."""

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

TEST_2026 = ('pmid:41534530', 'pmid:42069060', 'pmid:42048449')


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        root = value
        while root in self.parent and self.parent[root] != root:
            root = self.parent[root]
        self.parent.setdefault(root, root)
        while value in self.parent and self.parent[value] != root:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)

    def groups(self):
        groups = defaultdict(list)
        for name in list(self.parent):
            groups[self.find(name)].append(name)
        return {root: sorted(names) for root, names in groups.items()}


def reference_years(path):
    years = defaultdict(set)
    for line in path.open():
        raw = json.loads(line)['metadata']
        pmid = raw.get('Reference | PMID')
        reference = raw.get('Reference | IEDB IRI')
        date = str(raw.get('Reference | Date') or '')
        match = re.fullmatch(r'(19|20)\d{2}', date)
        if match:
            year = int(date)
            for study in ([f'pmid:{pmid}'] if pmid else []) + (
                [f'reference:{reference}'] if reference else []
            ):
                years[study].add(year)
    return years


def census_components(db, reference_path):
    components = Components()
    # Include aliases even for conflicting outcomes, so excluded duplicated records
    # cannot sever a whole-paper relationship and quietly permit leakage.
    for left, right in db.execute('SELECT r.study,a.study FROM rows r JOIN aliases a ON r.id=a.id'):
        components.union(left, right)
    for left, right in db.execute(
        'SELECT r.study,s.study FROM rows r JOIN source_studies s '
        'ON r.provenance_id=s.provenance_id'
    ):
        components.union(left, right)
    years = reference_years(reference_path)
    groups = components.groups()
    studies = {}
    for root, aliases in groups.items():
        identifier = 'study-component:' + hashlib.sha256('\n'.join(aliases).encode()).hexdigest()
        known_years = sorted({year for name in aliases for year in years.get(name, ())})
        studies[root] = dict(
            id=identifier,
            aliases=aliases,
            years=known_years,
            n=0,
            counts=Counter(),
            reported=defaultdict(Counter),
        )
    sql = '''SELECT study,endpoint,mhc_class,json_extract(data,'$.qualifier'),
             CASE WHEN endpoint IN ('qualitative_binding','elution_observation')
               THEN json_extract(data,'$.value') ELSE NULL END,COUNT(*)
             FROM rows WHERE excluded=0
             GROUP BY study,endpoint,mhc_class,json_extract(data,'$.qualifier'),
             CASE WHEN endpoint IN ('qualitative_binding','elution_observation')
               THEN json_extract(data,'$.value') ELSE NULL END'''
    # Physical magnitudes are never read: only qualifier counts and binary support.
    for study, endpoint, cls, qualifier, value, n in db.execute(sql):
        spec = studies[components.find(study)]
        spec['n'] += n
        if endpoint == 'elution_observation':
            label = int(value)
            spec['reported'][(study, cls)][label] += n
            if label == 1:
                spec['counts'][(endpoint, cls)] += n
        elif endpoint == 'qualitative_binding':
            spec['counts'][(endpoint, cls, str(int(value)))] += n
        elif qualifier == '=':
            spec['counts'][(endpoint, cls)] += n
    for spec in studies.values():
        for (_, cls), labels in spec.pop('reported').items():
            if min(labels[0], labels[1]):
                spec['counts'][('reported_ms_outcomes', cls)] += min(labels[0], labels[1])
    return components, studies


def target(key):
    if key[0] == 'elution_observation':
        return 512
    if key[0] in {'qualitative_binding', 'reported_ms_outcomes'}:
        return 32
    return 64


def select(studies, test_studies=TEST_2026, limit=12):
    by_alias = {alias: root for root, spec in studies.items() for alias in spec['aliases']}
    test = {by_alias[s] for s in test_studies if s in by_alias and studies[by_alias[s]]['n']}
    validation = set()
    initial_test = sorted(test)
    all_keys = {key for spec in studies.values() for key in spec['counts']}
    targets = {key: target(key) for key in all_keys}
    trace = []

    def choose(destination, reserve):
        for _ in range(limit):
            held = test | validation
            achieved = sum((studies[root]['counts'] for root in destination), Counter())
            remaining = {root: spec for root, spec in studies.items() if root not in held}
            totals = sum((spec['counts'] for spec in remaining.values()), Counter())
            supports = Counter(
                key for spec in remaining.values() for key, n in spec['counts'].items() if n
            )
            ranked = []
            for root, spec in remaining.items():
                if not spec['years'] or max(spec['years']) >= 2026:
                    continue
                if any(
                    totals[key] - n < min(64, totals[key] / 2) or supports[key] - 1 < reserve
                    for key, n in spec['counts'].items()
                    if n
                ):
                    continue
                gain = sum(
                    min(max(0, targets[key] - achieved[key]), n) / targets[key]
                    for key, n in spec['counts'].items()
                )
                if gain:
                    ranked.append((-(gain / math.log2(2 + spec['n'])), spec['id'], root, gain))
            if not ranked:
                break
            score, _, chosen, gain = min(ranked)
            destination.add(chosen)
            trace.append(
                dict(
                    split='test' if destination is test else 'validation',
                    component=studies[chosen]['id'],
                    aliases=studies[chosen]['aliases'],
                    gain=gain,
                    score=-score,
                )
            )

    choose(test, reserve=2)
    choose(validation, reserve=1)
    assignments = {
        root: 'test' if root in test else 'validation' if root in validation else 'train'
        for root in studies
    }
    receipt = dict(
        prespecified_2026_studies=list(test_studies),
        missing_2026_studies=[
            s for s in test_studies if s not in by_alias or not studies[by_alias[s]]['n']
        ],
        initial_2026_components=[studies[r]['id'] for r in initial_test],
        trace=trace,
        targets={':'.join(k): n for k, n in sorted(targets.items())},
        components=[
            dict(
                **{k: v for k, v in spec.items() if k != 'counts'},
                counts={':'.join(k): n for k, n in sorted(spec['counts'].items())},
                split=assignments[root],
            )
            for root, spec in sorted(studies.items())
        ],
        policy=(
            'whole study components; metadata-only greedy coverage; '
            'exact peptide embargo test > validation > train; no pretrained weights'
        ),
    )
    return assignments, receipt
