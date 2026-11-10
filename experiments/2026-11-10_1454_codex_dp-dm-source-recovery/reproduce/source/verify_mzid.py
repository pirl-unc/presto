"""Independently replay every recovered PSM against original XML using lxml."""

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from lxml import etree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--extraction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.perf_counter()
    rows = defaultdict(dict)
    with gzip.open(args.extraction / "source-psms.jsonl.gz", "rt") as src:
        for line in src:
            row = json.loads(line)
            key = row["identification_item"]["id"]
            assert key not in rows[row["source_file"]]
            rows[row["source_file"]][key] = row
    files = json.loads((args.extraction / "source-files.json").read_text())
    totals = Counter()
    groups = defaultdict(set)
    checks = 0
    assert set(rows) == {r["file"] for r in files}
    for info in files:
        path = args.sources / info["file"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["compressed_sha256"]
        with gzip.open(path) as stream:
            tree = etree.parse(stream)
        ns = {"p": tree.getroot().nsmap[None]}
        peptide = {e.get("id"): e for e in tree.xpath("//p:Peptide", namespaces=ns)}
        evidence = {
            e.get("id"): e for e in tree.xpath("//p:PeptideEvidence", namespaces=ns)
        }
        proteins = {e.get("id"): e for e in tree.xpath("//p:DBSequence", namespaces=ns)}
        sample = tree.xpath(
            "//p:AnalysisSampleCollection/p:Sample/@name", namespaces=ns
        )
        assert len(sample) == 1
        original = tree.xpath("//p:SpectrumIdentificationItem", namespaces=ns)
        assert len(original) == len(rows[info["file"]])
        observed_candidates = 0
        for element in original:
            row = rows[info["file"]][element.get("id")]
            assert dict(element.attrib) == row["identification_item"]
            assert dict(element.getparent().attrib) == row["identification_result"]
            assert row["source_sha256"] == info["compressed_sha256"]
            assert row["sample"]["source_sample_name"] == sample[0]
            pept = peptide[element.get("peptide_ref")]
            sequence = pept.xpath("p:PeptideSequence/text()", namespaces=ns)[0]
            assert row["peptide"] == sequence
            mods = pept.xpath("p:Modification", namespaces=ns)
            assert len(mods) == len(row["modifications"])
            for mod, saved in zip(mods, row["modifications"]):
                assert dict(mod.attrib) == saved["attributes"]
                assert [dict(cv.attrib) for cv in mod] == saved["parameters"]
            subs = pept.xpath("p:SubstitutionModification", namespaces=ns)
            assert [dict(e.attrib) for e in subs] == row["substitution_modifications"]
            refs = element.xpath(
                "p:PeptideEvidenceRef/@peptideEvidence_ref", namespaces=ns
            )
            assert len(refs) == len(row["protein_mappings"])
            decoys = set()
            for ref, saved in zip(refs, row["protein_mappings"]):
                ev = evidence[ref]
                decoys.add(ev.get("isDecoy"))
                assert dict(ev.attrib) == saved["evidence"]
                assert (
                    dict(proteins[ev.get("dBSequence_ref")].attrib) == saved["protein"]
                )
            assert [
                dict(e.attrib) for e in element.xpath("p:cvParam", namespaces=ns)
            ] == row["scores"]
            target = decoys == {"false"}
            unmodified = len(mods) + len(subs) == 0
            candidate = (
                target
                and unmodified
                and len(sequence) >= 10
                and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)
            )
            assert row["source_target_only"] == target
            assert row["reported_form"] == ("unmodified" if unmodified else "modified")
            assert row["sequence_and_form_candidate"] == candidate
            assert row["training_eligible"] is False and row["split"] == "unpartitioned"
            assert (
                row["confidence_status"]
                == "unqualified_mzid_flag_not_verified_1pct_fdr"
            )
            if candidate:
                groups[row["sample"]["biological_sample"]].add(sequence)
            totals.update(
                {
                    "n_source_psms": 1,
                    "n_modified_psms": int(not unmodified),
                    "n_target_only_psms": int(target),
                    "n_pass_threshold_true_psms": int(
                        element.get("passThreshold") == "true"
                    ),
                    "n_sequence_form_candidate_psms": int(candidate),
                }
            )
            observed_candidates += candidate
            checks += 1
        assert observed_candidates == info["n_sequence_form_candidate_psms"]
    summary = json.loads((args.extraction / "summary.json").read_text())
    assert dict(totals) == summary["counts"]
    for row in summary["biological_samples"]:
        assert (
            len(groups[row["biological_sample"]])
            == row["n_distinct_sequence_form_candidate_peptides"]
        )
    result = {
        "status": "passed",
        "n_original_files": len(files),
        "n_psms_replayed": checks,
        "checks": [
            "source hashes",
            "complete PSM and result attributes",
            "peptide sequences/forms",
            "original protein evidence and decoys",
            "score fields",
            "original sample names",
            "group counts",
            "no training eligibility",
        ],
        "seconds": time.perf_counter() - start,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
