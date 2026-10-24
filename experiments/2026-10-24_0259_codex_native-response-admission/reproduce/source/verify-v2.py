"""Independent source, reference, graph and target checks; never a source writer."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import resource
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text(value):
    return None if value is None else str(value).strip() or None


def normal(value):
    return " ".join(str(value or "").lower().split())


def identity(raw):
    if not text(raw):
        return None, [], "unidentified"
    body = text(raw)
    tentative = body.startswith("(") and body.endswith(")")
    body = body[1:-1].strip() if tentative else body
    match = re.fullmatch(r"(Ac-)?((?:Mox|[ACDEFGHIKLMNPQRSTVWY])+)", body)
    if not match:
        return None, [], "unresolved_annotation"
    tokens = re.findall(r"Mox|[ACDEFGHIKLMNPQRSTVWY]", match[2])
    mods = (
        [{"name": "N_terminal_acetylation", "position": 0, "source_token": "Ac-"}]
        if match[1]
        else []
    )
    mods += [
        {"name": "oxidation", "position": i, "source_token": "Mox"}
        for i, t in enumerate(tokens, 1)
        if t == "Mox"
    ]
    return (
        "".join("M" if t == "Mox" else t for t in tokens),
        mods,
        "tentative" if tentative else "reported_sequence",
    )


def matcher(patterns):
    """Aho-Corasick with overlapping and suffix matches, unlike prefix-index producer."""
    edges, failure, output = [{}], [0], [[]]
    for pattern in patterns:
        node = 0
        for char in pattern:
            if char not in edges[node]:
                edges[node][char] = len(edges)
                edges.append({})
                failure.append(0)
                output.append([])
            node = edges[node][char]
        output[node].append(pattern)
    queue = deque(edges[0].values())
    while queue:
        node = queue.popleft()
        for char, child in edges[node].items():
            queue.append(child)
            ancestor = failure[node]
            while ancestor and char not in edges[ancestor]:
                ancestor = failure[ancestor]
            failure[child] = edges[ancestor].get(char, 0)
            output[child].extend(output[failure[child]])

    def scan(sequence):
        node = 0
        for end, char in enumerate(sequence):
            while node and char not in edges[node]:
                node = failure[node]
            node = edges[node].get(char, 0)
            for pattern in output[node]:
                yield pattern, end - len(pattern) + 2  # 1-based

    return scan


def main():
    started = time.perf_counter()
    target = FAMILY / "results/independent-verification.json"
    assert not target.exists(), "Never overwrite verification evidence"
    config = read(FAMILY / "reproduce/launch.json")
    recipe = Path(config["recipe_directory"])
    manifest = read(recipe / "manifest.json")
    materialization = read(FAMILY / "results/materialization.json")
    assert sha(recipe / "manifest.json") == materialization["recipe_manifest_sha256"]
    for name, digest in manifest["files"].items():
        assert sha(recipe / name) == digest["sha256"], name
        assert (recipe / name).stat().st_size == digest["size_bytes"], name
    tables = read(recipe / "source_tables.json.gz")
    import pyarrow.parquet as pq

    for table, relative in {
        "source_rows": "source_rows",
        "measurements": "measurements",
        "lineage": "lineage",
        "contexts": "protocol_context",
    }.items():
        assert (
            pq.read_table(
                Path(config["source_package"]) / "tmab" / (relative + ".parquet")
            ).to_pylist()
            == tables[table]
        )
    rows = {r["source_record_id"]: r for r in tables["source_rows"]}
    measurements = {r["measurement_id"]: r for r in tables["measurements"]}
    contexts = {r["context_id"]: r for r in tables["contexts"]}
    lineage = {r["measurement_id"]: r for r in tables["lineage"]}
    assert len(rows) == len(tables["source_rows"])
    assert len(measurements) == len(tables["measurements"]) == len(lineage)
    source_by_row = defaultdict(list)
    for m in measurements.values():
        source_by_row[m["source_record_id"]].append(m)
    assertions = read(recipe / "assertions.json.gz")
    assert {a["source_record_id"] for a in assertions} == rows.keys()
    sequences = set()
    for a in assertions:
        row = rows[a["source_record_id"]]
        values = json.loads(row["source_row_json"])
        seq, mods, status = identity(row["source_sequence_raw"])
        assert (a["peptide_sequence"], a["sequence_identity_status"]) == (seq, status)
        if seq:
            sequences.add(seq)
        if row["source_row_disposition"] == "primary_ratio_panel":
            is_dasgupta = row["source_file"] == "dasgupta_s1.xlsx"
            assert a["parent_description_raw"] == text(values[6])
            assert a["gene_name_raw"] == (text(values[7]) if is_dasgupta else None)
            assert a["n_flank_raw"] == text(values[9 if is_dasgupta else 8])
            assert a["c_flank_raw"] == text(values[11 if is_dasgupta else 10])
        for m in source_by_row[row["source_record_id"]]:
            assert (m["peptide_sequence"], m["sequence_identity_status"]) == (seq, status)
            assert json.loads(m["peptide_modifications_json"]) == mods

    # Match every pattern against every full reference record, retaining alternatives.
    toy_patterns = ["A", "AA", "AAA", "ACA", "CA", "ACACA"]
    toy_sequence = "AAAACACAAAA"
    naive = sorted(
        (p, i + 1)
        for p in toy_patterns
        for i in range(len(toy_sequence))
        if toy_sequence.startswith(p, i)
    )
    assert sorted(matcher(toy_patterns)(toy_sequence)) == naive
    proteins = {}
    header = None
    parts = []

    def save():
        if header is not None:
            accession = header.split("|")[1]
            assert accession not in proteins and re.search(r"(?:^| )OX=9606(?: |$)", header)
            proteins[accession] = "".join(parts)

    for line in (recipe / "model_reference.fasta").read_text().splitlines():
        if line.startswith(">"):
            save()
            header, parts = line[1:], []
        else:
            parts.append(line.strip())
    save()
    scan = matcher(sorted(sequences))
    found = defaultdict(set)
    for accession, sequence in proteins.items():
        for peptide, start in scan(sequence):
            found[peptide].add((accession, start))
    observed_occurrences = read(recipe / "occurrences.json.gz")
    actual = {
        (r["peptide_sequence"], r["reference_accession"], r["start_1based"])
        for r in observed_occurrences
    }
    expected = {(s, a, p) for s, origins in found.items() for a, p in origins}
    assert actual == expected and len(actual) == len(observed_occurrences)
    used_parents = read(recipe / "parents.json.gz")
    assert {p["accession"] for p in used_parents} == {a for _, a, _ in expected}
    hashes = {a: hashlib.sha256(s.encode()).hexdigest() for a, s in proteins.items()}
    for p in used_parents:
        assert p["sequence"] == proteins[p["accession"]]
        assert p["sequence_sha256"] == hashes[p["accession"]]
    for r in observed_occurrences:
        s, start = proteins[r["reference_accession"]], r["start_1based"]
        end = start + len(r["peptide_sequence"]) - 1
        assert r["end_1based"] == end
        assert r["n_flank"] == (s[start - 2] if start > 1 else "-")
        assert r["c_flank"] == (s[end] if end < len(s) else "-")
        assert r["reference_sequence_sha256"] == hashes[r["reference_accession"]]

    # Bipartite graph traversal independently reconstructs all-source components.
    claims = defaultdict(set)
    row_claims = defaultdict(set)

    def add(row_id, claim):
        claims[claim].add(row_id)
        row_claims[row_id].add(claim)

    for a in assertions:
        r, seq = a["source_record_id"], a["peptide_sequence"]
        if seq:
            add(r, ("sequence", seq))
            for accession, _ in found[seq]:
                add(r, ("reference", hashes[accession]))
            for field in ("gene_name_raw", "parent_description_raw"):
                value = normal(a[field])
                if value and value not in {"unknown", "unidentified", "n/a"}:
                    add(r, (field, value))
    candidates = defaultdict(set)
    for line in lineage.values():
        candidate = line["candidate_reuse_group_id"]
        if candidate:
            add(line["source_record_id"], ("reuse", candidate))
            candidates[candidate].add(line["source_record_id"])
    remaining, visited_claims, membership = set(rows), set(), {}
    n_components = 0
    while remaining:
        todo, members = [next(iter(remaining))], set()
        while todo:
            row_id = todo.pop()
            if row_id in members:
                continue
            members.add(row_id)
            for claim in row_claims[row_id] - visited_claims:
                visited_claims.add(claim)
                todo.extend(claims[claim] - members)
        remaining -= members
        component = key(sorted(members))
        bucket = (
            int(hashlib.sha256(("tmab_native_response_v1" + component).encode()).hexdigest(), 16)
            % 20
        )
        split = "train" if bucket < 14 else "val" if bucket < 17 else "test"
        for row_id in members:
            membership[row_id] = (component, split)
        n_components += 1
    components = read(recipe / "components.json.gz")
    assert len(components) == len(rows)
    assert {
        r["source_record_id"]: (r["component_id"], r["split"]) for r in components
    } == membership

    def condition(m):
        c = contexts[m["context_id"]]
        return [
            normal(c["cell_line_name"]),
            normal(c["treatment_label_raw"]),
            c["treatment_name"],
            c["treatment_dose"],
            c["treatment_dose_unit"],
            c["duration_hours"],
        ]

    conflict = set()
    for candidate, ids in candidates.items():
        labels = {normal(rows[r]["source_experiment_raw"]) for r in ids}
        signatures = {key(condition(m)) for r in ids for m in source_by_row[r]}
        if len(labels) != 1 or len(signatures) != 1:
            conflict.add(candidate)
    admitted, reasons, expected_units = {}, Counter(), defaultdict(set)
    dispositions = {d["measurement_id"]: d for d in read(recipe / "dispositions.json.gz")}
    assert dispositions.keys() == measurements.keys()
    for mid, m in measurements.items():
        _, _, status = identity(m["peptide_sequence_raw"])
        value = m["quantitative_value"]
        if m["ratio_role"] != "treated_vs_control":
            reason = "deferred_control_or_derived_average"
        elif status != "reported_sequence":
            reason = "deferred_" + status
        elif value is None or not math.isfinite(value):
            reason = "deferred_nonquantifiable"
        elif value <= 0:
            reason = "deferred_nonpositive_ratio"
        elif m["censoring_status"] not in {"exact", "source_declared_bound"}:
            reason = "deferred_nonquantifiable"
        elif lineage[mid]["candidate_reuse_group_id"] in conflict:
            reason = "deferred_conflicting_possible_reuse_context"
        else:
            reason = None
        d = dispositions[mid]
        assert d["reason"] == reason
        assert d["status"] == ("admitted" if reason is None else "deferred")
        assert (d["component_id"], d["split"]) == membership[m["source_record_id"]]
        for name in (
            "quantitative_raw",
            "quantitative_value",
            "measurement_inequality",
            "censoring_status",
        ):
            assert d[name] == m[name]
        if reason:
            reasons[reason] += 1
        else:
            admitted[mid] = m
            expected_units[m["lineage_group_id"]].add(mid)

    examples = read(recipe / "examples.json.gz")
    conditions, parents, forms = defaultdict(set), defaultdict(set), defaultdict(set)
    for g in examples:
        conditions[g["split"]].add(g["condition_id"])
        parents[(g["split"], g["condition_id"])].add(g["component_id"])
        forms[(g["split"], g["condition_id"], g["component_id"])].add(g["response_group_id"])
    seen, units, proxy_splits = set(), {}, defaultdict(set)
    split_weights, obs_censor = Counter(), Counter()
    for g in examples:
        seq = g["peptide_sequence"]
        assert g["condition_id"] == key(g["condition"])
        assert g["response_group_id"] == key([[seq, g["modifications"]], g["condition"]])
        assert g["n_reference_occurrences"] == len(found[seq])
        assert g["n_reference_sequences"] == len({hashes[a] for a, _ in found[seq]})
        assert g["full_parent_model_supported"] == (
            bool(found[seq]) and all(set(proteins[a]) <= AA for a, _ in found[seq])
        )
        weight = (
            1
            / len(conditions[g["split"]])
            / len(parents[(g["split"], g["condition_id"])])
            / len(forms[(g["split"], g["condition_id"], g["component_id"])])
        )
        assert math.isclose(g["group_weight"], weight, rel_tol=1e-14)
        contexts_seen, proxies_seen = set(), set()
        for o in g["observations"]:
            assert o["observation_id"] not in units
            units[o["observation_id"]] = set(o["measurement_ids"])
            assert (
                len(o["measurement_ids"]) == len(units[o["observation_id"]]) == o["n_source_slots"]
            )
            assert math.isclose(o["weight"], weight / len(g["observations"]), rel_tol=1e-14)
            split_weights[g["split"]] += o["weight"]
            obs_censor[o["censoring_status"]] += 1
            contributors = []
            for mid in o["measurement_ids"]:
                assert mid not in seen
                seen.add(mid)
                m = admitted[mid]
                contributors.append(m)
                assert m["lineage_group_id"] == o["observation_id"]
                assert (
                    m["peptide_sequence"],
                    json.loads(m["peptide_modifications_json"]),
                    condition(m),
                ) == (seq, g["modifications"], g["condition"])
                assert (g["component_id"], g["split"]) == membership[m["source_record_id"]]
                assert o["quantitative_value"] == m["quantitative_value"]
                assert o["target_log10_ratio"] == math.log10(m["quantitative_value"])
                assert o["inequality"] == m["measurement_inequality"]
                assert o["censoring_status"] == m["censoring_status"]
                assert dispositions[mid]["response_group_id"] == g["response_group_id"]
                contexts_seen.add(m["context_id"])
                proxies_seen.add(m["shared_control_group_id"])
            assert o["source_record_ids"] == sorted({m["source_record_id"] for m in contributors})
            assert o["charge_labels_raw"] == sorted(
                {m["charge_raw"] for m in contributors if m["charge_raw"]}
            )
            assert o["replicate_labels"] == sorted({m["replicate_label"] for m in contributors})
        assert g["context_ids"] == sorted(contexts_seen)
        assert g["shared_control_proxy_ids"] == sorted(proxies_seen)
        for proxy in proxies_seen:
            proxy_splits[proxy].add(g["split"])
    assert seen == admitted.keys()
    assert units == expected_units
    assert all(math.isclose(w, 1, abs_tol=1e-12) for w in split_weights.values())
    summary = read(recipe / "summary.json.gz")
    assert summary["n_components"] == n_components
    assert summary["n_conflicting_reuse_groups"] == len(conflict)
    assert summary["n_control_proxies_crossing_splits"] == sum(
        len(s) > 1 for s in proxy_splits.values()
    )
    assert summary["defer_reason_counts"] == dict(reasons)
    assert summary["observations_by_censoring_status"] == dict(obs_censor)
    receipt = {
        "status": "passed",
        "recipe_manifest_sha256": sha(recipe / "manifest.json"),
        "source_manifest_sha256": sha(recipe / "source_manifest.json"),
        "verifier_sha256": sha(__file__),
        "n_source_rows": len(rows),
        "n_measurement_slots": len(measurements),
        "n_reference_proteins": len(proteins),
        "n_reference_occurrences": len(expected),
        "n_split_components": n_components,
        "n_effective_observations": len(units),
        "n_response_groups": len(examples),
        "n_admitted_source_slots": len(seen),
        "split_weight_sums": dict(split_weights),
        "n_shared_control_proxies_crossing_splits": summary["n_control_proxies_crossing_splits"],
        "specimen_independence": "unresolved",
        "biological_training": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    assert receipt["seconds"] < 180 and receipt["peak_rss_bytes"] < 4 * 1024**3
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
