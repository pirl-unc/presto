"""Independently verify FASTA origins, graph components and source artifacts."""

import csv
import gzip
import hashlib
import json
import resource
import shutil
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from Bio import SeqIO

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
raw = Path(config["raw_directory"])
results = ROOT / "results"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            value.update(block)
    return value.hexdigest()


for asset in config["input_assets"]:
    path = Path(asset["path"])
    assert path.stat().st_size == asset["size_bytes"] and sha(path) == asset["sha256"]
receipts = [json.loads(p.read_text()) for p in raw.glob("*.receipt.json")]
assert len(receipts) == 4 and all(r["access_status"] == "retrieved_and_length_verified" for r in receipts)
for receipt in receipts:
    path = Path(receipt["path"])
    assert path.stat().st_size == receipt["size_bytes"] and sha(path) == receipt["sha256"]
for name in ("member-verification.json", "evidence-member-verification.json"):
    report = json.loads((results / name).read_text())
    assert report["passed"] and report["independent_zipfile_bytes_and_crcs_equal"]
    for member in report["members"]:
        path = raw / "processed-members" / Path(member["name"]).name
        assert path.stat().st_size == member["uncompressed_bytes"] and sha(path) == member["sha256"]
assert json.loads((results / "independent-reconciliation-verification.json").read_text())["passed"]

reference = json.loads((results / "reference-records.json").read_text())
sequences = {}
for index, record in enumerate(SeqIO.parse(raw / "HUMAN_NS3_Cyclin.fasta", "fasta")):
    expected = reference[index]
    sequence = str(record.seq)
    assert record.description == expected["header"]
    assert record.id.split("|")[1] == expected["accession"]
    assert hashlib.sha256(sequence.encode()).hexdigest() == expected["sequence_sha256"]
    assert len(sequence) == expected["n_residues"]
    sequences[index] = sequence
assert len(sequences) == len(reference) == 73113
hash_by_accession = {r["accession"]:r["sequence_sha256"] for r in reference}
with (raw / "processed-members/peptides.txt").open() as stream:
    peptides = {p["id"]:p for p in csv.DictReader(stream, delimiter="\t")}
adjacency = defaultdict(set)
candidate_parents, candidate_kinds = defaultdict(set), defaultdict(set)
candidate_counts, literal_counts, query_counts = Counter(), Counter(), Counter()


def link(left, right):
    adjacency[left].add(right)
    adjacency[right].add(left)


with gzip.open(results / "reference-origins.csv.gz", "rt", newline="") as stream:
    for origin in csv.DictReader(stream):
        peptide = peptides[origin["peptide_id"]]
        reference_id = int(origin["reference_record_id"])
        parent = reference[reference_id]
        start, end = int(origin["start_1based"])-1, int(origin["end_1based"])
        span = sequences[reference_id][start:end]
        assert 0 <= start < end <= len(sequences[reference_id])
        assert span.replace("L","I") == peptide["Sequence"].replace("L","I")
        assert (origin["match_type"] == "literal") == (span == peptide["Sequence"])
        assert origin["parent_accession"] == parent["accession"]
        assert origin["parent_sequence_sha256"] == parent["sequence_sha256"]
        assert origin["reference_disposition"] == parent["disposition"]
        assert (origin["source_declared_accession"] == "True") == (parent["accession"] in peptide["Proteins"].split(";"))
        pid = peptide["id"]
        candidate_parents[pid].add(parent["sequence_sha256"])
        candidate_kinds[pid].add(parent["disposition"])
        candidate_counts[pid] += 1
        literal_counts[pid] += origin["match_type"] == "literal"
        query_counts[peptide["Sequence"].replace("L","I")] += 1
        link("peptide:" + pid, "sequence:" + parent["sequence_sha256"])
independent = json.loads((results / "independent-occurrence-counts.json").read_text())
assert len(independent) == len(peptides)
assert all(query_counts[query] == count for query,count in independent.items())
for pid, peptide in peptides.items():
    adjacency["peptide:" + pid]
    for token in filter(None, peptide["Proteins"].split(";")):
        node = "sequence:" + hash_by_accession[token] if token in hash_by_accession else "source_parent:" + token
        link("peptide:" + pid, node)
group_by_node, unseen = {}, set(adjacency)
while unseen:
    start = unseen.pop()
    nodes, pending = {start}, [start]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in nodes:
                nodes.add(neighbor)
                unseen.discard(neighbor)
                pending.append(neighbor)
    group = hashlib.sha256("\n".join(sorted(nodes)).encode()).hexdigest()
    group_by_node.update({node:group for node in nodes})
with (results / "peptide-dependency-groups.csv").open() as stream:
    published_groups = {r["peptide_id"]:r["dependency_group"] for r in csv.DictReader(stream)}
assert set(published_groups) == set(peptides)
assert all(group_by_node["peptide:" + pid] == group for pid,group in published_groups.items())
support = json.loads((results / "descriptive-support.json").read_text())
strict = {pid:p for pid,p in peptides.items() if not p["Reverse"] and not p["Potential contaminant"]
          and p["PEP"] and Decimal(p["PEP"]) <= Decimal("0.01") and len(candidate_parents[pid]) == 1
          and candidate_kinds[pid] == {"reported_human_endogenous_reference"}
          and candidate_counts[pid] == literal_counts[pid]}
assert len(strict) == support["strict_origin_dispositions"]["strict_origin_support"]
for stratum in support["descriptive_presence_strata"]:
    groups = defaultdict(list)
    minimum = stratum["minimum_positive_acquisitions_per_condition"]
    for pid,p in strict.items():
        if all(sum(Decimal(p["Intensity " + arm + str(i)])>0 for i in (1,2,3)) >= minimum for arm in ("Cont","KD")):
            groups[next(iter(candidate_parents[pid]))].append(pid)
    multi = [pids for pids in groups.values() if len(pids)>=2]
    assert stratum["n_strict_peptides_with_presence"] == sum(map(len, groups.values()))
    assert stratum["n_parent_sequences_with_two_or_more_peptides"] == len(multi)
    assert stratum["n_peptides_in_multi_peptide_parents"] == sum(map(len,multi))
    assert stratum["n_independent_dependency_components"] == len({published_groups[pid] for pids in multi for pid in pids})
    assert stratum["n_possible_single_reference_contrasts"] == sum(len(pids)-1 for pids in multi)

assets = [dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(raw.rglob("*")) if path.is_file()]
with (results / "source-assets.json").open("x") as output:
    output.write(json.dumps(dict(assets=assets),indent=2)+"\n")
assert sum(a["size_bytes"] for a in assets) <= config["limits"]["total_raw_bytes"]
assert shutil.disk_usage(ROOT).free >= config["limits"]["disk_floor_bytes"]
assert time.monotonic()-started < config["limits"]["stage_seconds"]
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < config["limits"]["rss_bytes"]
report = dict(passed=True,n_original_inputs_verified=len(config["input_assets"]),
              n_download_receipts_verified=len(receipts),n_original_processed_members_verified=5,
              n_reference_records_independently_reparsed=len(reference),
              n_origin_spans_independently_verified=sum(candidate_counts.values()),
              n_peptide_dependency_groups_independently_replayed=len(published_groups),
              n_descriptive_support_strata_replayed=3,
              source_qualified_for_canonical_admission=True,canonical_hitlist_admission_complete=False,
              model_fitted=False,apc_concentration_goal_achieved=False,
              original_archive_full_sha1_verified=False,
              seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              argv=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (results / "source-closure.json").open("x") as output:
    output.write(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
