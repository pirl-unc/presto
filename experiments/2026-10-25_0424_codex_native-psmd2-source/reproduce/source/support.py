"""Quantity-presence support only; no condition effect or model is computed."""

import csv
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
raw = Path(config["raw_directory"])
with (raw / "processed-members/peptides.txt").open() as stream:
    peptides = list(csv.DictReader(stream, delimiter="\t"))
support = json.loads((ROOT / "results/reference-support.json").read_text())
records = json.loads((ROOT / "results/reference-records.json").read_text())
hash_by_accession = {r["accession"]: r["sequence_sha256"] for r in records}
parent = {}


def find(node):
    parent.setdefault(node, node)
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(left, right):
    parent[find(left)] = find(right)


for peptide in peptides:
    node = "peptide:" + peptide["id"]
    find(node)
    candidates = support.get(peptide["id"], {})
    for value in candidates.get("parent_sequences", []):
        union(node, "sequence:" + value)
    for token in filter(None, peptide["Proteins"].split(";")):
        key = "sequence:" + hash_by_accession[token] if token in hash_by_accession else "source_parent:" + token
        union(node, key)
components = defaultdict(list)
for node in parent:
    components[find(node)].append(node)
component_id = {root: hashlib.sha256("\n".join(sorted(nodes)).encode()).hexdigest() for root,nodes in components.items()}
peptide_group = {p["id"]:component_id[find("peptide:" + p["id"])] for p in peptides}
sizes = Counter(peptide_group.values())
with (ROOT / "results/peptide-dependency-groups.csv").open("x", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["peptide_id", "dependency_group"])
    writer.writerows(sorted(peptide_group.items(), key=lambda x:int(x[0])))

strict, dispositions = [], Counter()
for peptide in peptides:
    info = support.get(peptide["id"])
    reason = (
        "source_reverse_or_contaminant" if peptide["Reverse"] or peptide["Potential contaminant"] else
        "pep_above_001_or_missing" if not peptide["PEP"] or Decimal(peptide["PEP"]) > Decimal("0.01") else
        "no_reference_origin" if info is None else
        "multiple_distinct_parent_sequences" if len(info["parent_sequences"]) != 1 else
        "nonendogenous_reference_possible" if info["dispositions"] != ["reported_human_endogenous_reference"] else
        "il_only_alternative_span" if info["n_spans"] != info["n_literal_spans"] else
        "strict_origin_support"
    )
    dispositions[reason] += 1
    if reason == "strict_origin_support":
        strict.append(peptide)
strata = []
for minimum in (1, 2, 3):
    by_parent = defaultdict(list)
    for peptide in strict:
        control = sum(Decimal(peptide["Intensity Cont" + str(i)]) > 0 for i in (1,2,3))
        knockdown = sum(Decimal(peptide["Intensity KD" + str(i)]) > 0 for i in (1,2,3))
        if min(control, knockdown) >= minimum:
            by_parent[support[peptide["id"]]["parent_sequences"][0]].append(peptide["id"])
    multi = {parent:pids for parent,pids in by_parent.items() if len(pids)>=2}
    eligible = [pid for pids in multi.values() for pid in pids]
    strata.append(dict(minimum_positive_acquisitions_per_condition=minimum,
                      n_strict_peptides_with_presence=sum(map(len,by_parent.values())),
                      n_parent_sequences_with_two_or_more_peptides=len(multi),
                      n_peptides_in_multi_peptide_parents=len(eligible),
                      n_independent_dependency_components=len({peptide_group[p] for p in eligible}),
                      n_possible_single_reference_contrasts=sum(len(pids)-1 for pids in multi.values())))
result = dict(n_all_source_peptides=len(peptides),n_all_source_dependency_components=len(sizes),
              largest_component_n_peptides=max(sizes.values()),strict_origin_dispositions=dict(dispositions),
              descriptive_presence_strata=strata,
              biological_replicate_independence_established=False,
              numerical_contrasts_reference_choices_splits_or_models_computed=False,
              prospective_model_eligibility_not_fixed=True,
              seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (ROOT / "results/descriptive-support.json").open("x") as output:
    output.write(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
