"""Recover source PSMs and sample identities without asserting training eligibility."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parameters(element, ns):
    return [dict(e.attrib) for e in element.findall("./m:cvParam", ns)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--sdrf", type=Path, required=True)
    parser.add_argument("--constructs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.inventory.read_text())
    files = sorted(
        (r for r in inventory if r["fileName"].endswith(".mzid.gz")),
        key=lambda r: r["fileName"],
    )
    assert len(files) == 32 and len({r["fileName"] for r in files}) == 32
    constructs = json.loads(args.constructs.read_text())
    with args.sdrf.open() as stream:
        sdrf_rows = list(csv.DictReader(stream, delimiter="\t"))
    sdrf = {r["comment[data file]"].removesuffix(".raw"): r for r in sdrf_rows}
    assert len(sdrf_rows) == len(sdrf) == 32
    all_samples, file_records, counts, forms = [], [], Counter(), Counter()
    group_peptides = defaultdict(set)
    group_counts = defaultdict(Counter)
    out_path = args.output / "source-psms.jsonl.gz"
    with gzip.open(out_path, "wt") as output:
        for item in files:
            path = args.sources / item["fileName"]
            binary = path.read_bytes()
            assert hashlib.sha1(binary).hexdigest() == item["checksum"], path
            payload = gzip.decompress(binary)
            assert len(payload) == item["fileSizeBytes"], path
            sha = hashlib.sha256(binary).hexdigest()
            root = ET.fromstring(payload)
            ns = {"m": root.tag.split("}")[0][1:]}
            samples = root.findall(".//m:AnalysisSampleCollection/m:Sample", ns)
            assert len(samples) == 1
            sample = samples[0]
            source_name = sample.attrib["name"]
            match = re.fullmatch(r"(DP401|DP402|DP10)(_DM)?_([AB])([123])", source_name)
            control = re.fullmatch(r"No_DP_control_([AB])([123])", source_name)
            assert bool(match) != bool(control), source_name
            if match:
                construct, dm, biological, technical = match.groups()
                pair = constructs["constructs"][construct]
                role = "engineered_dp_capture"
                dm_state = "present" if dm else "absent"
                arm = construct + ("_DM" if dm else "")
            else:
                biological, technical = control.groups()
                pair = {"alpha": None, "beta": None}
                dm_state = "unknown"
                role = "no_dp_capture_control"
                arm = "No_DP_control"
            run = path.name.removesuffix(".mzid.gz")
            community = sdrf[run]
            assert community["source name"] == source_name[:-1]
            assert community["comment[technical replicate]"] == technical
            assert community["characteristics[biological replicate]"] == str(
                "AB".index(biological) + 1
            )
            if pair["beta"] is not None:
                assert community["characteristics[mhc typing]"] == pair["beta"]
            spectra = root.findall(".//m:SpectraData", ns)
            assert len(spectra) == 1 and spectra[0].get("name") == run
            mapping = {
                "file": path.name,
                "sha256": sha,
                "run": run,
                "source_sample_name": source_name,
                "biological_sample": source_name[:-1],
                "biological_replicate": biological,
                "technical_replicate": int(technical),
                "arm": arm,
                "capture_role": role,
                "mhc_alpha": pair["alpha"],
                "mhc_beta": pair["beta"],
                "dm_state": dm_state,
                "cell_line": "HeLa",
                "construct_evidence": constructs["source"],
                "sample_evidence": "original mzIdentML AnalysisSampleCollection",
                "community_sdrf_agrees": True,
            }
            all_samples.append(mapping)
            db = {
                e.attrib["id"]: dict(e.attrib)
                for e in root.findall(".//m:DBSequence", ns)
            }
            peptides = {e.attrib["id"]: e for e in root.findall(".//m:Peptide", ns)}
            evidence = {
                e.attrib["id"]: dict(e.attrib)
                for e in root.findall(".//m:PeptideEvidence", ns)
            }
            file_counts = Counter()
            psm_ids = set()
            for result in root.findall(".//m:SpectrumIdentificationResult", ns):
                assert result.get("spectraData_ref") == spectra[0].get("id")
                for psm in result.findall("./m:SpectrumIdentificationItem", ns):
                    pid = psm.attrib["id"]
                    assert pid not in psm_ids
                    psm_ids.add(pid)
                    assert psm.get("sample_ref") == sample.attrib["id"]
                    peptide = peptides[psm.attrib["peptide_ref"]]
                    sequence = peptide.findtext("./m:PeptideSequence", namespaces=ns)
                    modifications = [
                        {"attributes": dict(e.attrib), "parameters": parameters(e, ns)}
                        for e in peptide.findall("./m:Modification", ns)
                    ]
                    substitutions = [
                        dict(e.attrib)
                        for e in peptide.findall("./m:SubstitutionModification", ns)
                    ]
                    protein_mappings = []
                    for e in psm.findall("./m:PeptideEvidenceRef", ns):
                        ev = evidence[e.attrib["peptideEvidence_ref"]]
                        assert ev["peptide_ref"] == peptide.attrib["id"]
                        protein_mappings.append(
                            {"evidence": ev, "protein": db[ev["dBSequence_ref"]]}
                        )
                    assert protein_mappings
                    decoy_states = {
                        p["evidence"].get("isDecoy") for p in protein_mappings
                    }
                    target_only = decoy_states == {"false"}
                    unmodified = not modifications and not substitutions
                    candidate = (
                        target_only
                        and unmodified
                        and len(sequence) > 9
                        and set(sequence) <= AA
                    )
                    row = {
                        "source_file": path.name,
                        "source_sha256": sha,
                        "sample": mapping,
                        "identification_result": dict(result.attrib),
                        "identification_item": dict(psm.attrib),
                        "peptide_id": peptide.attrib["id"],
                        "peptide": sequence,
                        "modifications": modifications,
                        "substitution_modifications": substitutions,
                        "protein_mappings": protein_mappings,
                        "scores": parameters(psm, ns),
                        "reported_form": "unmodified" if unmodified else "modified",
                        "source_target_only": target_only,
                        "sequence_and_form_candidate": candidate,
                        "confidence_status": "unqualified_mzid_flag_not_verified_1pct_fdr",
                        "training_eligible": False,
                        "split": "unpartitioned",
                    }
                    output.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    file_counts["n_source_psms"] += 1
                    file_counts["n_modified_psms"] += not unmodified
                    file_counts["n_target_only_psms"] += target_only
                    file_counts["n_pass_threshold_true_psms"] += (
                        psm.get("passThreshold") == "true"
                    )
                    file_counts["n_sequence_form_candidate_psms"] += candidate
                    forms["modified" if not unmodified else "unmodified"] += 1
                    if candidate:
                        group_peptides[mapping["biological_sample"]].add(sequence)
                    group_counts[mapping["biological_sample"]].update(
                        {
                            "n_source_psms": 1,
                            "n_sequence_form_candidate_psms": int(candidate),
                        }
                    )
            protocols = []
            for p in root.findall(".//m:SpectrumIdentificationProtocol", ns):
                protocols.append(
                    {
                        "attributes": dict(p.attrib),
                        "threshold": [
                            dict(e.attrib)
                            for e in p.findall("./m:Threshold/m:cvParam", ns)
                        ],
                        "search_modifications": [
                            ET.tostring(e, encoding="unicode")
                            for e in p.findall(
                                "./m:ModificationParams/m:SearchModification", ns
                            )
                        ],
                    }
                )
            file_records.append(
                {
                    "file": path.name,
                    "compressed_sha256": sha,
                    "compressed_sha1": hashlib.sha1(binary).hexdigest(),
                    "compressed_bytes": len(binary),
                    "uncompressed_bytes": len(payload),
                    "protocols": protocols,
                    **dict(file_counts),
                }
            )
            counts.update(file_counts)
    summary = {
        "dataset": "PXD017154",
        "n_files": len(files),
        "n_source_samples": len({r["biological_sample"] for r in all_samples}),
        "n_arms": len({r["arm"] for r in all_samples}),
        "counts": dict(counts),
        "forms": dict(forms),
        "biological_samples": [
            {
                **next(r for r in all_samples if r["biological_sample"] == name),
                **dict(group_counts[name]),
                "n_distinct_sequence_form_candidate_peptides": len(values),
            }
            for name, values in sorted(group_peptides.items())
        ],
        "n_training_eligible_rows": 0,
        "confidence_qualification_pending": True,
        "source_psms_path": str(out_path),
        "source_psms_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    dump(args.output / "source-samples.json", all_samples)
    dump(args.output / "source-files.json", file_records)
    dump(args.output / "summary.json", summary)
    print(
        json.dumps(
            {"n_files": len(files), **dict(counts), "seconds": summary["seconds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
