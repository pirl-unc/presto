"""All deposited-reference origins, including separate I/L ambiguity evidence."""

import csv
import gzip
import hashlib
import io
import json
import resource
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
RAW = Path(CONFIG["raw_directory"])
RESULTS = ROOT / "results"
fasta = RAW / "HUMAN_NS3_Cyclin.fasta"
assert hashlib.sha1(fasta.read_bytes()).hexdigest() == "ebfe21f8f804ceb7679d4c41c001fde25cea2a96"
with (RAW / "processed-members/peptides.txt").open() as stream:
    peptides = list(csv.DictReader(stream, delimiter="\t"))


def guard():
    assert time.monotonic() - started < CONFIG["limits"]["stage_seconds"]
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < CONFIG["limits"]["rss_bytes"]
    assert shutil.disk_usage(ROOT).free >= CONFIG["limits"]["disk_floor_bytes"]


def read_fasta():
    header, parts = None, []
    with fasta.open() as stream:
        for line in stream:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:].strip(), []
            else:
                parts.append(line.strip())
        if header is not None:
            yield header, "".join(parts)


records = []
sequence_by_record = {}
for index, (header, sequence) in enumerate(read_fasta()):
    assert sequence and sequence.isalpha() and sequence == sequence.upper()
    fields = header.split("|", 2)
    assert len(fields) == 3
    accession = fields[1]
    # Explicit deposited construct provenance overrides its misleading taxonomy.
    reporter = accession == "P22222" and fields[2].startswith("NS3CCNB1_HUMAN ")
    disposition = "engineered_reporter_not_predictive_target" if reporter else (
        "reported_human_endogenous_reference" if "OX=9606 " in header else "other_reference_unresolved")
    records.append(dict(reference_record_id=index, accession=accession, header=header,
                        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                        n_residues=len(sequence), disposition=disposition))
    sequence_by_record[index] = sequence
assert sum(r["disposition"] == "engineered_reporter_not_predictive_target" for r in records) == 1

queries = defaultdict(list)
for peptide in peptides:
    queries[peptide["Sequence"].replace("L", "I")].append(peptide)
prefixes = defaultdict(list)
for query in queries:
    assert len(query) >= 5
    prefixes[query[:5]].append(query)

occurrence_counts = Counter()
per_peptide = defaultdict(lambda: dict(n_spans=0, parent_records=set(), parent_sequences=set(),
                                     parent_accessions=set(), dispositions=set(), n_literal_spans=0))
origin_path = RESULTS / "reference-origins.csv.gz"
fields = ["peptide_id", "reference_record_id", "parent_accession", "parent_sequence_sha256",
          "start_1based", "end_1based", "match_type", "reference_disposition", "source_declared_accession"]
with origin_path.open("xb") as fileobj:
    with gzip.GzipFile(filename="", mode="wb", fileobj=fileobj, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in records:
                if record["reference_record_id"] % 1000 == 0:
                    guard()
                original = sequence_by_record[record["reference_record_id"]]
                normalized = original.replace("L", "I")
                for start in range(len(normalized) - 4):
                    for query in prefixes.get(normalized[start:start+5], ()):
                        if not normalized.startswith(query, start):
                            continue
                        occurrence_counts[query] += 1
                        for peptide in queries[query]:
                            literal = original[start:start+len(query)] == peptide["Sequence"]
                            writer.writerow(dict(peptide_id=peptide["id"], reference_record_id=record["reference_record_id"],
                                parent_accession=record["accession"],parent_sequence_sha256=record["sequence_sha256"],
                                start_1based=start+1,end_1based=start+len(query),match_type="literal" if literal else "isoleucine_leucine_equivalent",
                                reference_disposition=record["disposition"],source_declared_accession=record["accession"] in peptide["Proteins"].split(";")))
                            support = per_peptide[peptide["id"]]
                            support["n_spans"] += 1
                            support["n_literal_spans"] += literal
                            support["parent_records"].add(record["reference_record_id"])
                            support["parent_sequences"].add(record["sequence_sha256"])
                            support["parent_accessions"].add(record["accession"])
                            support["dispositions"].add(record["disposition"])

index_seconds = time.monotonic() - started
with (RESULTS / "reference-records.json").open("x") as output:
    output.write(json.dumps(records, separators=(",", ":")) + "\n")
with (RESULTS / "reference-support.json").open("x") as output:
    output.write(json.dumps({k:{field:sorted(value) if isinstance(value,set) else value for field,value in row.items()}
                            for k,row in per_peptide.items()}, separators=(",", ":")) + "\n")

# Independent completeness check: C-level whole-reference searches for each
# normalized query, separated by a character absent from peptide sequences.
joined = "#".join(sequence_by_record[r["reference_record_id"]].replace("L", "I") for r in records)
independent_counts = {}
for index, query in enumerate(queries):
    if index % 50 == 0:
        guard()
    position, count = 0, 0
    while True:
        position = joined.find(query, position)
        if position == -1:
            break
        count += 1
        position += 1
    independent_counts[query] = count
    assert count == occurrence_counts[query], "Independent full-reference count differs"
with (RESULTS / "independent-occurrence-counts.json").open("x") as output:
    output.write(json.dumps(independent_counts, separators=(",", ":")) + "\n")

reference_accessions = {r["accession"] for r in records}
declared = {p for row in peptides for p in row["Proteins"].split(";") if p}
absent = sorted(declared - reference_accessions)
unrepresented = [p["id"] for p in peptides if p["id"] not in per_peptide]
summary = dict(
    n_reference_records=len(records),n_unique_reference_accessions=len(reference_accessions),
    n_unique_reference_sequences=len({r["sequence_sha256"] for r in records}),
    n_reference_residues=sum(r["n_residues"] for r in records),
    reference_dispositions=dict(Counter(r["disposition"] for r in records)),
    n_peptides=len(peptides),n_unique_il_normalized_queries=len(queries),
    n_peptides_with_reference_origins=len(per_peptide),n_peptides_without_reference_origins=len(unrepresented),
    unrepresented_peptide_ids=unrepresented,
    n_peptide_span_records=sum(s["n_spans"] for s in per_peptide.values()),
    n_literal_peptide_span_records=sum(s["n_literal_spans"] for s in per_peptide.values()),
    n_peptides_with_one_distinct_parent_sequence=sum(len(s["parent_sequences"])==1 for s in per_peptide.values()),
    n_peptides_with_multiple_distinct_parent_sequences=sum(len(s["parent_sequences"])>1 for s in per_peptide.values()),
    n_peptides_with_reporter_origin=sum("engineered_reporter_not_predictive_target" in s["dispositions"] for s in per_peptide.values()),
    n_declared_parent_tokens=len(declared),n_declared_tokens_absent_from_deposited_reference=len(absent),
    absent_declared_tokens=absent,
    n_peptides_with_unreported_reference_accession=sum(bool(s["parent_accessions"]-set(peptides[int(pid)]["Proteins"].split(";"))) for pid,s in per_peptide.items()),
    independent_all_query_occurrence_counts_equal=True,
    index_seconds=index_seconds,seconds=time.monotonic()-started,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    origins_sha256=hashlib.sha256(origin_path.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=sys.argv,
    training_eligibility_or_model_effects_computed=False,
)
with (RESULTS / "reference-summary.json").open("x") as output:
    output.write(json.dumps(summary, indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k not in ("unrepresented_peptide_ids","absent_declared_tokens")}, indent=2))
