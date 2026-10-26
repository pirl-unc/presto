"""Independent source, catalog, context, disposition and contributor replay.

Does not import the admission adapter or HumanMHCCatalog.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
OUTPUT = Path("/private/tmp/presto-mhc2-20261026/ms-admitted/attempt01")
RESULTS = FAMILY / "results/admission-attempt01"
PROFILE = FAMILY / "results/input-profile"
AA = set("ACDEFGHIKLMNPQRSTVWY")
UNIQUE = {
    "curated_sample_label",
    "elution_conditions",
    "allele_exact",
    "discriminated",
    "single_sample_pmid",
}
PREFIXES = {
    "condition_knockout_genes": "knockout",
    "condition_knockdown_genes": "knockdown",
    "condition_overexpression_genes": "overexpression",
    "condition_transfection": "transfection",
    "condition_transduction": "transduction",
    "condition_drugs": "drug",
}
UNKNOWN = {"unknown", "unspecified", "not_reported", "unreported", "not_applicable"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def records(path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=2048, columns=columns, use_threads=False
    ):
        yield from batch.to_pylist()


def class_ii_records(path, counter):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, use_threads=False):
        indices = [
            i
            for i, (cls, ms) in enumerate(
                zip(
                    batch.column("mhc_class").to_pylist(),
                    batch.column("is_ms_observation").to_pylist(),
                )
            )
            if cls == "II" and ms is True
        ]
        for position, row in zip(
            indices, batch.take(pa.array(indices, type=pa.int64())).to_pylist()
        ):
            yield counter[0] + position, row
        counter[0] += len(batch)


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def condition_supported(row):
    return bool(
        row["sample_attribution"] in UNIQUE
        and row["condition_id"]
        and row["condition_status"] == "annotated"
        and row["condition_evidence"] in {"primary_source", "curated_text"}
        and (row["condition_evidence"] != "primary_source" or row["condition_reference"])
    )


def tokens(value):
    return set(part.strip() for part in (value or "").split(";") if part.strip())


def main():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    destination = RESULTS / "independent-verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    launch = json.loads((FAMILY / "reproduce/admission-launch-attempt01.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    for entry in [launch[k] for k in ("source", "catalog", "contributors")] + summary["artifacts"]:
        assert sha(Path(entry["path"])) == entry["sha256"], entry["path"]
    aliases = defaultdict(set)
    with Path(launch["catalog"]["path"]).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not (
                row["source"] == "imgt"
                and row["species"] == "Homo sapiens"
                and row["source_id"]
                and row["two_field_allele"].startswith("HomoSapiens-")
                and row["is_functional"] == "True"
                and all(row[k] == "False" for k in ("is_null", "is_questionable", "is_pseudogene"))
                and row["groove_status"] == "ok"
                and not row["groove_flags"]
                and row["mhc_class"] == "II"
                and row["chain"] in {"alpha", "beta"}
            ):
                continue
            slot = "groove1" if row["chain"] == "alpha" else "groove2"
            other = "groove2" if slot == "groove1" else "groove1"
            domain = row[slot]
            if (
                row[other]
                or not (60 <= len(domain) <= 130)
                or set(domain) - AA
                or domain not in row["sequence"]
                or row[slot + "_len"] != str(len(domain))
            ):
                continue
            name = row["two_field_allele"].replace("HomoSapiens-", "HLA-", 1)
            aliases[name].add((row["chain"], domain))

    @lru_cache(maxsize=4096)
    def resolve(value):
        result = defaultdict(set)
        if not value:
            return None
        names = [x.strip() for x in value.replace(",", ";").split(";") if x.strip()]
        for name in names:
            parts = name.split("/")
            if len(parts) != 2 or not name.startswith("HLA-"):
                return None
            parts = [x if x.startswith("HLA-") else "HLA-" + x for x in parts]
            if any(len(aliases[x]) != 1 for x in parts):
                return None
            domains = dict(next(iter(aliases[x])) for x in parts)
            if set(domains) != {"alpha", "beta"}:
                return None
            result[(domains["alpha"], domains["beta"])].add(name)
        return dict(result) or None

    context_groups = json.loads((PROFILE / "context-groups.json").read_text())
    vocabulary = defaultdict(lambda: defaultdict(set))
    for row in context_groups:
        if condition_supported(row):
            for field in (*PREFIXES, "condition_cytokines"):
                vocabulary[row["pmid"]][field].update(tokens(row[field]) - UNKNOWN - {"none"})
    serialized = {
        pmid: {field: sorted(names) for field, names in fields.items()}
        for pmid, fields in vocabulary.items()
    }
    assert serialized == json.loads((RESULTS / "study-factor-vocabulary.json").read_text())
    expected_profile = {}
    actual_profile = {}
    for label in ("molecular", "context"):
        entries = json.loads((PROFILE / f"{label}-groups.json").read_text())
        fields = sorted(set(entries[0]) - {"rows"})
        expected_profile[label] = (
            fields,
            Counter({tuple(row[k] for k in fields): row["rows"] for row in entries}),
        )
        actual_profile[label] = Counter()
    expected_categories = json.loads((PROFILE / "categorical-fields.json").read_text())
    category_counts = {name: Counter() for name in expected_categories}
    dispositions = iter(records(OUTPUT / "dispositions.parquet"))
    counts, studies, factor_counts, subtypes = Counter(), Counter(), Counter(), Counter()
    peptide_set, retained_ids, provenance_ids = set(), set(), set()
    checked = selected_rows = 0
    source_count = [0]
    with (OUTPUT / "observations.jsonl").open() as observations:
        for ordinal, raw in class_ii_records(launch["source"]["path"], source_count):
            assert raw["mhc_class"] == "II" and raw["is_ms_observation"] is True
            selected_rows += 1
            receipt = next(dispositions)
            assert (
                receipt["source_ordinal"] == ordinal
                and receipt["evidence_row_id"] == raw["evidence_row_id"]
            )
            counts[(receipt["disposition"], receipt["reason"])] += 1
            for label, (fields, _) in expected_profile.items():
                actual_profile[label][
                    tuple(None if raw[k] is None else str(raw[k]) for k in fields)
                ] += 1
            for name in category_counts:
                category_counts[name][raw[name]] += 1
            scoped = raw["mhc_allele_provenance"] == "exact" or (
                raw["mhc_allele_provenance"] in {"sample_allele_match", "sample_locus_match"}
                and raw["sample_attribution"] in UNIQUE
            )
            eligible = bool(
                raw["assay_modality"] == "ms"
                and raw["evidence_kind"] == "ms"
                and raw["ms_outcome_status"] in {"positive_result", "curated_ligand_table"}
                and raw["mhc_species"] == "Homo sapiens"
                and raw["has_ptm"] is False
                and not raw["peptide_modifications"]
                and raw["is_non_peptide_ligand"] is False
                and raw["is_potential_contaminant"] is not True
                and isinstance(raw["peptide"], str)
                and 9 <= len(raw["peptide"]) <= 512
                and not set(raw["peptide"]) - AA
                and raw["provenance_status"] == "indexed"
                and all(
                    isinstance(raw[k], str) and raw[k].strip()
                    for k in ("evidence_row_id", "evidence_source_id", "provenance_id")
                )
                and raw["restriction_evidence"] != "predicted"
                and scoped
            )
            pairs = resolve(raw["mhc_allele_set"]) if eligible else None
            assert (receipt["disposition"] == "admitted") == bool(eligible and pairs)
            if not pairs:
                continue
            row = json.loads(next(observations))
            assert row["provenance"]["raw_record"] == raw
            assert row["provenance"]["source_ordinal"] == ordinal
            assert row["provenance"]["source_sha256"] == launch["source"]["sha256"]
            assert (
                row["observation_id"] == raw["evidence_row_id"] and row["peptide"] == raw["peptide"]
            )
            assert (
                row["value"] == 1
                and row["unit"] == "binary"
                and row["endpoint"] == "elution_observation"
            )
            assert row["split"] == "predict" and row["group_id"] == "unpartitioned"
            observed_pairs = {
                (pair["groove1"], pair["groove2"]): set(pair["provenance"]["reported_candidates"])
                for pair in row["mhc"]
            }
            assert observed_pairs == pairs and len(row["mhc"]) == len(pairs)
            assert all(
                pair["mhc_class"] == "II"
                and pair["species1"] == pair["species2"] == "human"
                and pair["provenance"]["partner_imputation"] is False
                for pair in row["mhc"]
            )
            supported = condition_supported(raw)
            context = row["context"]
            assert context["provenance"]["condition_admitted"] == supported
            expected_states = {"interventions": {}, "cytokines": {}}
            if supported:
                for field in (*PREFIXES, "condition_cytokines"):
                    stated = tokens(raw[field])
                    assert "none" not in stated or stated == {"none"}
                    absent = stated == {"none"}
                    names = vocabulary[raw["pmid"]][field] if absent else stated - UNKNOWN
                    for name in names:
                        family = "cytokines" if field == "condition_cytokines" else "interventions"
                        label = name if family == "cytokines" else PREFIXES[field] + ":" + name
                        expected_states[family][label] = "absent" if absent else "present"
                        claim = context[family][label]
                        assert (
                            claim["evidence"]["source_field"] == field
                            and claim["evidence"]["source_value"] == raw[field]
                        )
                        assert claim["evidence"]["reference"] == raw["condition_reference"]
            for family, expected in expected_states.items():
                assert {name: value["state"] for name, value in context[family].items()} == expected
                for name, state in expected.items():
                    factor_counts[(family, name, state)] += 1
            expected_cell = (
                raw["cell_type"]
                if supported and raw["cell_type"] and ";" not in raw["cell_type"]
                else None
            )
            assert context["cell_type"] == expected_cell
            assert context["components"] == context["processing_pathways"] == {}
            assert context["source_n"] is None and context["source_c"] is None
            assert row["observation_id"] not in retained_ids
            retained_ids.add(row["observation_id"])
            provenance_ids.add(raw["provenance_id"])
            studies[raw["pmid"]] += 1
            peptide_set.add(raw["peptide"])
            kinds = {name.removeprefix("HLA-")[:2] for names in pairs.values() for name in names}
            subtypes["+".join(sorted(kinds))] += 1
            checked += 1
            if checked % 10000 == 0 and peak_rss() > 3 * 1024**3:
                raise RuntimeError("verification exceeded 3-GiB RSS limit")
        assert next(dispositions, None) is None and next(observations, None) is None
    source_rows = source_count[0]
    assert source_rows == summary["source_rows"] and selected_rows == summary["class_ii_ms_rows"]
    assert checked == summary["admitted_rows"]
    assert dict(studies) == json.loads((RESULTS / "by-study.json").read_text())
    assert counts == Counter(
        {(row["disposition"], row["reason"]): row["rows"] for row in summary["dispositions"]}
    )
    for label, (_, expected) in expected_profile.items():
        assert actual_profile[label] == expected, label
    for name, entries in expected_categories.items():
        assert category_counts[name] == Counter({row["value"]: row["rows"] for row in entries}), (
            name
        )
    link_file = OUTPUT / "contributor-links.parquet"
    columns = pq.ParquetFile(link_file).schema_arrow.names
    retained_links = iter(records(link_file))
    link_counts, pmids = Counter(), defaultdict(set)
    unknown, ambiguous = set(), set()
    all_links = 0
    for row in records(launch["contributors"]["path"], columns):
        all_links += 1
        key = row["provenance_id"]
        if key not in provenance_ids:
            continue
        assert row == next(retained_links)
        link_counts[key] += 1
        if row["relationship_status"] == "overlap_unresolved":
            ambiguous.add(key)
        if str(row["original_pmid"] or "").isdigit() and int(row["original_pmid"]) > 0:
            pmids[key].add(str(row["original_pmid"]))
        else:
            unknown.add(key)
    assert next(retained_links, None) is None
    assert (
        set(link_counts) == provenance_ids and all_links == summary["all_source_contributor_links"]
    )
    assert sum(link_counts.values()) == summary["retained_contributor_links"]
    for row in records(OUTPUT / "contributor-summary.parquet"):
        key = row["provenance_id"]
        assert row["links"] == link_counts.pop(key)
        assert row["original_pmids"] == sorted(pmids[key])
        assert row["overlap_unresolved"] == (key in ambiguous)
        assert row["study_status"] == ("unknown" if key in unknown else "resolved")
    assert not link_counts
    result = {
        "status": "passed",
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)),
        "source_rows_verified": source_rows,
        "class_ii_ms_dispositions_verified": selected_rows,
        "positive_observations_verified": checked,
        "distinct_peptides": len(peptide_set),
        "retained_contributor_links_verified": summary["retained_contributor_links"],
        "molecular_subtypes": dict(subtypes),
        "context_factor_support": [
            {"family": key[0], "name": key[1], "state": key[2], "rows": value}
            for key, value in sorted(factor_counts.items())
        ],
        "source_profile_replay": "all groups and categorical values match",
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss(),
        "predictive_qualification": False,
    }
    assert result["peak_rss_bytes"] <= 3 * 1024**3
    destination.write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(__file__, FAMILY / "reproduce/source" / Path(__file__).name)
    signal.alarm(0)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "context_factor_support"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
