"""Score held-out Presto predictions against external binding-affinity tools.

The harness is deliberately split into pure functions so the rules can be unit
tested on tiny synthetic data without any tool binary:

* ``extract_records``       per-example dump -> flat rows
* ``map_allele``            prospective allele-name rules (mhcgnomes)
* ``annotate_eligibility``  one exclusion reason per row, shared by all methods
* ``parse_*_output``        NetMHCpan / NetMHCIIpan text parsers
* ``unit_metrics`` ...      within class x endpoint x detector x study metrics
* ``pooled_metrics``        mean-of-study with study-level cluster bootstrap

The rules are pre-registered in
``experiments/agents/claude/plans/2026-10-31_external-netmhc-benchmark.md``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from presto.training.holdout_eval import (
    binding_threshold_metrics,
    pearson,
    spearman,
)

BA_ENDPOINTS = ("ic50", "kd", "kd_via_ic50", "kd_via_ec50")
QUALITATIVE_ENDPOINT = "qualitative_binding"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
LENGTH_WINDOW = {"I": (8, 15), "II": (9, 30)}
THRESHOLD_NM = 500.0
MIN_UNIT_ROWS = 20
MAX_STUDY_ROWS = 20_000
MAX_WORKERS = 4
BOOTSTRAP_SEED = 20261031
# Tools whose predictions must exist for a row of each MHC class.
CLASS_TOOLS = {"I": ("netmhcpan41", "mhcflurry"), "II": ("netmhciipan43",)}
ALL_TOOLS = ("netmhcpan41", "netmhciipan43", "mhcflurry")
EXCLUSION_ORDER = (
    "multi_or_ambiguous",
    "bad_residue",
    "length",
    "unsupported_allele",
    "tool_failure",
)


# ----------------------------------------------------------------------------
# dump extraction
# ----------------------------------------------------------------------------


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _detector_from_raw(prov: Mapping) -> str:
    """Detector from the IEDB assay-method string (``prep/mechanism/detector``) if the dump
    carries no structured ``assay_method`` (the curated-retrain family)."""
    text = ((prov.get("raw_record") or {}).get("assay_method") or "").lower()
    for detector in ("fluorescence", "radioactivity"):
        if text.endswith(detector):
            return detector
    return "none"


def extract_record(row: Mapping) -> Optional[dict]:
    """Flatten one dump row, or return None when it is outside BA/qualitative scope."""
    endpoint = row["endpoint"]
    if endpoint not in BA_ENDPOINTS and endpoint != QUALITATIVE_ENDPOINT:
        return None
    obs = row["observation"]
    prov = obs.get("provenance") or {}
    mhc = obs.get("mhc") or []
    catalog = [a for m in mhc for a in (m.get("provenance") or {}).get("catalog_alleles", [])]
    method = obs.get("assay_method") or {}
    curation = (prov.get("measurement_curation") or {}).get("status")
    raw_value = row.get("raw_value")
    raw_nm = float(raw_value) if row.get("raw_unit") == "nM" and raw_value else float("nan")
    target = row.get("target")
    return {
        "observation_id": row["observation_id"],
        "peptide": row["peptide"],
        "mhc_class": (row.get("mhc_classes") or [None])[0],
        "endpoint": endpoint,
        "detector": method.get("detector") or _detector_from_raw(prov),
        "study_id": prov.get("study_id") or "unknown",
        "qualifier": row.get("qualifier") or "=",
        "raw_value": raw_value,
        "raw_nM": raw_nm,
        "target_log10": float(target) if endpoint in BA_ENDPOINTS else float("nan"),
        "presto": float(row["prediction"]),
        "n_mhc_entries": len(mhc),
        "catalog_alleles": "|".join(catalog),
        "bound_source": "inferred" if str(curation or "").startswith("inferred") else "explicit",
    }


def extract_records(path: os.PathLike) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = extract_record(json.loads(line))
            if record is not None:
                rows.append(record)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# allele rules
# ----------------------------------------------------------------------------


def _parse_hla(name: str) -> Optional[Tuple[str, str, str]]:
    """(gene, field1, field2) for a human four-digit allele, else None."""
    import mhcgnomes

    try:
        parsed = mhcgnomes.parse(name)
    except Exception:
        return None
    if type(parsed).__name__ != "Allele":
        return None
    if parsed.species.mhc_prefix != "HLA" or len(parsed.allele_fields) != 2:
        return None
    if parsed.mutations or parsed.gene.mutations:
        return None
    first, second = parsed.allele_fields
    if not (first.isdigit() and second.isdigit()):
        return None
    return parsed.gene.name, first, second


def map_allele(mhc_class: str, catalog_alleles: Sequence[str], n_mhc_entries: int = 1):
    """Tool-specific allele names, or None when the row is multi-allelic/ambiguous.

    Returns ``{"netmhcpan41"|"netmhciipan43": name, "mhcflurry": name}`` (keys present
    only for tools applicable to the class).
    """
    if n_mhc_entries != 1 or not catalog_alleles:
        return None
    parsed = [_parse_hla(a) for a in catalog_alleles]
    if any(p is None for p in parsed):
        return None
    if mhc_class == "I":
        if len(parsed) != 1 or parsed[0][0] not in {"A", "B", "C"}:
            return None
        gene, f1, f2 = parsed[0]
        return {"netmhcpan41": f"HLA-{gene}{f1}{f2}", "mhcflurry": f"HLA-{gene}*{f1}:{f2}"}
    if mhc_class == "II":
        if len(parsed) != 2:
            return None
        by_gene = {p[0]: p for p in parsed}
        if len(by_gene) != 2:
            return None
        if "DRA" in by_gene:
            (other,) = [g for g in by_gene if g != "DRA"]
            if by_gene["DRA"][1:] != ("01", "01") or other not in {"DRB1", "DRB3", "DRB4", "DRB5"}:
                return None
            return {"netmhciipan43": f"{other}_{by_gene[other][1]}{by_gene[other][2]}"}
        for alpha, beta in (("DQA1", "DQB1"), ("DPA1", "DPB1")):
            if set(by_gene) == {alpha, beta}:
                a, b = by_gene[alpha], by_gene[beta]
                return {"netmhciipan43": f"HLA-{alpha}{a[1]}{a[2]}-{beta}{b[1]}{b[2]}"}
        return None
    return None


def subsample_per_study(
    frame: pd.DataFrame, cap: int = MAX_STUDY_ROWS
) -> Tuple[pd.DataFrame, List[dict]]:
    """Deterministic cap per study by ascending SHA-256(observation_id)."""
    kept, log = [], []
    for study, group in frame.groupby("study_id", sort=True):
        if len(group) > cap:
            keys = group["observation_id"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
            group = group.loc[keys.sort_values(kind="mergesort").index[:cap]]
            log.append({"study_id": study, "kept": cap})
        kept.append(group)
    out = pd.concat(kept) if kept else frame
    return out, log


def annotate_eligibility(
    frame: pd.DataFrame, supported: Mapping[str, Iterable[str]]
) -> pd.DataFrame:
    """Add tool allele names and a single ``exclusion`` reason (None = eligible).

    ``supported`` maps tool name -> allele names that tool's own list accepts. A row
    unsupported by any tool applicable to its class is excluded for all methods.
    """
    supported_sets = {tool: set(names) for tool, names in supported.items()}
    out = frame.copy()
    for tool in ALL_TOOLS:
        out[f"allele_{tool}"] = None
    out["exclusion"] = None
    for i, row in out.iterrows():
        cls = row["mhc_class"]
        names = map_allele(
            cls, [a for a in row["catalog_alleles"].split("|") if a], row["n_mhc_entries"]
        )
        reason = None
        if names is None:
            reason = "multi_or_ambiguous"
        elif not set(row["peptide"]) <= STANDARD_AA:
            reason = "bad_residue"
        else:
            low, high = LENGTH_WINDOW[cls]
            if not low <= len(row["peptide"]) <= high:
                reason = "length"
            elif any(names[t] not in supported_sets[t] for t in CLASS_TOOLS[cls]):
                reason = "unsupported_allele"
        if names is not None:
            for tool, name in names.items():
                out.at[i, f"allele_{tool}"] = name
        out.at[i, "exclusion"] = reason
    return out


def apply_tool_failures(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop (label) rows missing a finite prediction from any applicable tool."""
    out = frame.copy()
    for cls, tools in CLASS_TOOLS.items():
        for tool in tools:
            missing = (
                (out["mhc_class"] == cls)
                & out["exclusion"].isna()
                & ~np.isfinite(out.get(f"pred_{tool}", pd.Series(np.nan, index=out.index)))
            )
            out.loc[missing, "exclusion"] = "tool_failure"
    return out


# ----------------------------------------------------------------------------
# tool execution and parsing
# ----------------------------------------------------------------------------


def _table_rows(text: str, affinity_column: str) -> Tuple[Dict[str, float], Optional[float]]:
    """peptide -> nM from NetMHCpan/NetMHCIIpan output, plus 'Distance to training data'."""
    result: Dict[str, float] = {}
    distance = None
    match = re.search(r"Distance to training data\s+([0-9.]+)", text)
    if match:
        distance = float(match.group(1))
    columns: Optional[List[str]] = None
    for line in text.splitlines():
        tokens = line.split()
        if affinity_column in tokens and "Peptide" in tokens:
            columns = tokens
            continue
        if columns is None or not tokens or tokens[0].startswith("-"):
            continue
        if not tokens[0].isdigit() or len(tokens) < len(columns) - 1:
            continue
        peptide = tokens[columns.index("Peptide")]
        result[peptide] = float(tokens[columns.index(affinity_column)])
    return result, distance


def parse_netmhcpan_output(text: str) -> Tuple[Dict[str, float], Optional[float]]:
    return _table_rows(text, "Aff(nM)")


def parse_netmhciipan_output(text: str) -> Tuple[Dict[str, float], Optional[float]]:
    return _table_rows(text, "Affinity(nM)")


def _tool_env(bundle_home: str, threads: int = 1) -> dict:
    env = dict(os.environ)
    env["NETMHC_BUNDLE_HOME"] = bundle_home
    tmp = env.get("NETMHC_BUNDLE_TMPDIR") or os.path.join(
        tempfile.gettempdir(), "netmhc-bundle-temp"
    )
    os.makedirs(tmp, exist_ok=True)
    env["NETMHC_BUNDLE_TMPDIR"] = tmp
    env["OMP_NUM_THREADS"] = str(threads)
    return env


def _run_bundle_job(binary: str, tool: str, allele: str, peptides: Sequence[str], bundle_home: str):
    column = "Aff(nM)" if tool == "netmhcpan41" else "Affinity(nM)"
    with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as handle:
        handle.write("\n".join(peptides) + "\n")
        path = handle.name
    try:
        if tool == "netmhcpan41":
            cmd = [binary, "-p", path, "-a", allele, "-BA"]
        else:
            cmd = [binary, "-inptype", "1", "-f", path, "-a", allele, "-BA"]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=_tool_env(bundle_home), check=False
        )
    finally:
        os.unlink(path)
    values, distance = _table_rows(proc.stdout, column)
    rows = [
        {
            "tool": tool,
            "allele": allele,
            "peptide": pep,
            "ic50_nM": values.get(pep, float("nan")),
            "allele_distance_to_training": distance,
        }
        for pep in peptides
    ]
    return rows, {"tool": tool, "allele": allele, "returncode": proc.returncode, "n": len(peptides)}


def run_bundle_tool(
    tool: str,
    pairs: pd.DataFrame,
    bundle_home: str,
    *,
    chunk: int = 2000,
    workers: int = MAX_WORKERS,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Run NetMHCpan-4.1 or NetMHCIIpan-4.3 over (allele, peptide) pairs, <=4 processes."""
    binary = {
        "netmhcpan41": f"{bundle_home}/bin/netMHCpan-4.1",
        "netmhciipan43": f"{bundle_home}/bin/netMHCIIpan-4.3",
    }[tool]
    jobs = []
    for allele, group in pairs.groupby("allele", sort=True):
        peptides = sorted(set(group["peptide"]))
        for start in range(0, len(peptides), chunk):
            jobs.append((allele, peptides[start : start + chunk]))
    rows, logs = [], []
    with ThreadPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
        futures = [
            pool.submit(_run_bundle_job, binary, tool, allele, peps, bundle_home)
            for allele, peps in jobs
        ]
        for future in futures:
            job_rows, log = future.result()
            rows.extend(job_rows)
            logs.append(log)
    return pd.DataFrame(rows), logs


def run_mhcflurry(pairs: pd.DataFrame, executable: str, workdir: str) -> pd.DataFrame:
    """Run ``mhcflurry-predict`` (affinity only) over (allele, peptide) pairs."""
    unique = pairs[["allele", "peptide"]].drop_duplicates().reset_index(drop=True)
    inp, out = os.path.join(workdir, "mhcflurry_in.csv"), os.path.join(workdir, "mhcflurry_out.csv")
    unique.to_csv(inp, index=False)
    env = dict(os.environ, OMP_NUM_THREADS=str(MAX_WORKERS))
    subprocess.run(
        [executable, inp, "--allele-column", "allele", "--peptide-column", "peptide",
         "--affinity-only", "--backend", "cpu", "--num-jobs", "0", "--out", out],
        check=True, env=env, capture_output=True, text=True,
    )  # fmt: skip
    result = pd.read_csv(out)
    return pd.DataFrame(
        {
            "tool": "mhcflurry",
            "allele": result["allele"],
            "peptide": result["peptide"],
            "ic50_nM": result["mhcflurry_affinity"],
            "allele_distance_to_training": np.nan,
        }
    )


def attach_tool_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Add ``pred_<tool>`` columns (log10 nM) by (tool allele name, peptide)."""
    out = frame.copy()
    for tool in ALL_TOOLS:
        sub = predictions[predictions["tool"] == tool]
        lookup = {(a, p): v for a, p, v in zip(sub["allele"], sub["peptide"], sub["ic50_nM"])}
        values = [
            lookup.get((a, p), float("nan")) if a is not None else float("nan")
            for a, p in zip(out[f"allele_{tool}"], out["peptide"])
        ]
        arr = np.asarray(values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            arr = np.where(arr > 0, np.log10(arr), np.nan)
        out[f"pred_{tool}"] = arr
    return out


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

UNIT_KEYS = ["mhc_class", "endpoint", "detector", "study_id"]
POOL_KEYS = ["mhc_class", "detector"]
METRICS = (
    "spearman", "pearson", "rmse", "accuracy", "balanced_accuracy",
    "precision", "recall", "f1", "auroc", "auprc",
)  # fmt: skip


def methods_for(mhc_class: str) -> List[str]:
    return ["presto", "train_mean"] + list(CLASS_TOOLS[mhc_class])


def _pred(group: pd.DataFrame, method: str) -> np.ndarray:
    return group[method if method in ("presto", "train_mean") else f"pred_{method}"].to_numpy(float)


def _one(truth: np.ndarray, raw: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out = {
        "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "spearman": spearman(truth, pred),
        "pearson": pearson(truth, pred),
    }
    out = {k: (float("nan") if v is None else v) for k, v in out.items()}
    binding = binding_threshold_metrics(
        truth, pred, np.zeros(len(truth)), threshold_nm=THRESHOLD_NM, raw_target_nm=raw
    )
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc"):
        out[key] = binding.get(key, float("nan"))
    return out


def _bootstrap_spearman_rmse(truth, pred, rng, draws):
    n = len(truth)
    idx = rng.integers(0, n, size=(draws, n))
    sp, rm = np.full(draws, np.nan), np.empty(draws)
    for b in range(draws):
        i = idx[b]
        rho = spearman(truth[i], pred[i])
        sp[b] = np.nan if rho is None else rho
        rm[b] = math.sqrt(float(np.mean((truth[i] - pred[i]) ** 2)))
    return sp, rm


def _ci(values: np.ndarray) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def unit_metrics(
    exact: pd.DataFrame, *, row_draws: int = 1000, seed: int = BOOTSTRAP_SEED
) -> pd.DataFrame:
    """Per class x endpoint x detector x study metrics for every method (exact rows)."""
    rows = []
    for key, group in exact.groupby(UNIT_KEYS, sort=True):
        cls = key[0]
        truth = group["target_log10"].to_numpy(float)
        raw = group["raw_nM"].to_numpy(float)
        for method in methods_for(cls):
            pred = _pred(group, method)
            record = dict(zip(UNIT_KEYS, key), method=method, n=len(group))
            record.update(_one(truth, raw, pred))
            if row_draws and len(group) >= MIN_UNIT_ROWS:
                sp, rm = _bootstrap_spearman_rmse(
                    truth, pred, np.random.default_rng(seed), row_draws
                )
                record["spearman_ci_lo"], record["spearman_ci_hi"] = _ci(sp)
                record["rmse_ci_lo"], record["rmse_ci_hi"] = _ci(rm)
            rows.append(record)
    return pd.DataFrame(rows)


def _fisher_mean(rho: np.ndarray, n: np.ndarray) -> float:
    ok = np.isfinite(rho) & (n > 3)
    if not ok.any():
        return float("nan")
    z = np.arctanh(np.clip(rho[ok], -0.999999, 0.999999))
    return float(np.tanh(np.sum((n[ok] - 3) * z) / np.sum(n[ok] - 3)))


def pooled_metrics(
    units: pd.DataFrame, *, draws: int = 2000, seed: int = BOOTSTRAP_SEED
) -> pd.DataFrame:
    """Within-study pooled metrics per class x detector (never across detectors).

    Each study contributes the mean of its (endpoint) units with n >= MIN_UNIT_ROWS.
    CI: percentile of the study-level cluster bootstrap. ``diff_vs_presto`` is always
    method minus Presto on paired draws (so for RMSE a negative value favours the method).
    """
    eligible = units[units["n"] >= MIN_UNIT_ROWS]
    rows = []
    for pool_key, pool in eligible.groupby(POOL_KEYS, sort=True):
        per_study = {}
        for method, mgroup in pool.groupby("method"):
            per_study[method] = mgroup.groupby("study_id")[list(METRICS)].mean()
        studies = sorted(pool["study_id"].unique())
        presto = per_study["presto"].reindex(studies)
        for method, table in per_study.items():
            table = table.reindex(studies)
            fisher_src = pool[pool["method"] == method]
            record_base = dict(zip(POOL_KEYS, pool_key), method=method)
            for metric in METRICS:
                a = table[metric].to_numpy(float)
                ok = np.isfinite(a)
                record = dict(record_base, metric=metric, n_studies=int(ok.sum()),
                              n_rows=int(fisher_src.loc[fisher_src["study_id"].isin(
                                  np.array(studies)[ok]), "n"].sum()))  # fmt: skip
                if ok.sum():
                    record["mean_of_study"] = float(np.mean(a[ok]))
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, ok.sum(), size=(draws, ok.sum()))
                    record["ci_lo"], record["ci_hi"] = _ci(a[ok][idx].mean(axis=1))
                    b = presto[metric].to_numpy(float)
                    both = ok & np.isfinite(b)
                    if method != "presto" and both.sum():
                        d = (a - b)[both]
                        idx = rng.integers(0, both.sum(), size=(draws, both.sum()))
                        record["diff_vs_presto"] = float(d.mean())
                        record["diff_ci_lo"], record["diff_ci_hi"] = _ci(d[idx].mean(axis=1))
                if metric == "spearman":
                    record["fisher_z_weighted"] = _fisher_mean(
                        fisher_src["spearman"].to_numpy(float), fisher_src["n"].to_numpy(float)
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def censored_metrics(censored: pd.DataFrame) -> pd.DataFrame:
    """Bound satisfaction by class x detector x study x direction x bound source."""
    frame = censored.copy()
    frame["direction"] = np.where(
        frame["qualifier"].isin([">", ">="]), "lower_bound", "upper_bound"
    )
    rows = []
    keys = ["mhc_class", "detector", "study_id", "endpoint", "direction", "bound_source"]
    for key, group in frame.groupby(keys, sort=True):
        bound = np.log10(group["raw_nM"].to_numpy(float))
        lower = key[4] == "lower_bound"
        for method in methods_for(key[0]):
            pred = _pred(group, method)
            violation = np.maximum(bound - pred, 0) if lower else np.maximum(pred - bound, 0)
            failed = violation > 0
            rows.append(
                dict(
                    zip(keys, key), method=method, n=len(group),
                    bound_threshold_median_nM=float(np.median(group["raw_nM"])),
                    satisfied=float(np.mean(~failed)),
                    n_violated=int(failed.sum()),
                    median_violation_log10=(
                        float(np.median(violation[failed])) if failed.any() else 0.0
                    ),
                    mean_violation_log10=float(np.mean(violation)),
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def qualitative_metrics(qualitative: pd.DataFrame) -> pd.DataFrame:
    """AUROC per class x study on literal Positive/Negative rows (identical rows)."""
    from presto.training.holdout_eval import auroc

    rows = []
    for key, group in qualitative.groupby(["mhc_class", "study_id"], sort=True):
        labels = (group["raw_value"].astype(float) > 0.5).astype(float).to_numpy()
        for method in methods_for(key[0]):
            if method == "train_mean":
                continue
            score = _pred(group, method)
            if method != "presto":
                score = -score  # lower nM = stronger binding
            value = auroc(labels, score)
            rows.append(
                dict(mhc_class=key[0], study_id=key[1], method=method, n=len(group),
                     n_pos=int(labels.sum()), n_neg=int(len(labels) - labels.sum()),
                     auroc=float("nan") if value is None else value)
            )  # fmt: skip
    return pd.DataFrame(rows)


def exclusion_census(frame: pd.DataFrame) -> pd.DataFrame:
    """Row counts by class, scope kind and exclusion reason."""
    kind = np.where(
        frame["endpoint"] == QUALITATIVE_ENDPOINT,
        "qualitative",
        np.where(frame["qualifier"] == "=", "exact_ba", "censored_ba"),
    )
    table = (
        frame.assign(kind=kind, exclusion=frame["exclusion"].fillna("eligible"))
        .groupby(["mhc_class", "kind", "exclusion"])
        .size()
        .rename("n")
        .reset_index()
    )
    return table


def add_train_mean(frame: pd.DataFrame, scales: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """TRAIN-only per-endpoint mean log10(nM) from the run's frozen scales."""
    out = frame.copy()
    out["train_mean"] = out["endpoint"].map(
        lambda e: float(scales[e]["mean"]) if e in scales else float("nan")
    )
    return out


def score_dump(
    frame: pd.DataFrame, *, row_draws: int = 1000, draws: int = 2000
) -> Dict[str, pd.DataFrame]:
    """All tables for one dump after eligibility and tool attachment."""
    eligible = frame[frame["exclusion"].isna()]
    ba = eligible[eligible["endpoint"].isin(BA_ENDPOINTS)]
    exact = ba[ba["qualifier"] == "="]
    units = unit_metrics(exact, row_draws=row_draws)
    return {
        "census": exclusion_census(frame),
        "units": units,
        "pooled": pooled_metrics(units, draws=draws),
        "censored": censored_metrics(ba[ba["qualifier"] != "="]),
        "qualitative": qualitative_metrics(eligible[eligible["endpoint"] == QUALITATIVE_ENDPOINT]),
    }


def tool_versions(bundle_home: str, mhcflurry_executable: str) -> dict:
    """Pin what we can read locally."""
    root = Path(bundle_home)
    info = {
        "netMHCpan": (root / "netMHCpan-4.1/data/version").read_text().strip(),
        "netMHCIIpan": (root / "netMHCIIpan-4.3/data/version").read_text().strip(),
    }
    proc = subprocess.run([mhcflurry_executable, "--version"], capture_output=True, text=True)
    info["mhcflurry"] = (proc.stdout or proc.stderr).strip()
    lic = root / "netMHCIIpan-4.3/netMHCIIpan-4.3_license.txt"
    info["netMHCIIpan_license_sha256"] = sha256_file(lic) if lic.exists() else None
    return info
