"""Prospective affinity comparisons with explicit assay, allele and tool identity.

The historical external_benchmark API stays frozen for old-result reproduction.
This module consumes physical-unit predictions and does not choose test cohorts.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from presto.data.assay_measurement import observation_method
from presto.data.modular import Observation, observation_target
from presto.training.modular_controls import assay_allele_key
from .external_benchmark import _one, _table_rows, _tool_env, map_allele, sha256_file

CELL = [
    "mhc_class",
    "endpoint",
    "source",
    "study",
    "preparation",
    "mechanism",
    "detector",
    "mhc_key",
]
AFFINITY = {"ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50"}


def affinity_record(record):
    """Preserve method/study provenance and reject mismatched physical target exports."""
    if record["endpoint"] not in AFFINITY:
        return None
    obs = Observation.from_dict(record["observation"])
    if obs.endpoint != record["endpoint"] or obs.observation_id != record["observation_id"]:
        raise ValueError("prediction/observation identity mismatch")
    target = observation_target(obs)
    if record.get("target_space") != "log10(nM)" or not math.isclose(
        target, record["target"], rel_tol=1e-7, abs_tol=1e-7
    ):
        raise ValueError("physical affinity target mismatch")
    prediction = float(record["prediction"])
    if not math.isfinite(prediction):
        raise ValueError("nonfinite Presto prediction")
    key, method = assay_allele_key(obs), observation_method(obs)
    classes = {m.class_override or m.mhc_class for m in obs.mhc}
    cls = next(iter(classes)) if len(classes) == 1 else None
    alleles = sorted({a for m in obs.mhc for a in m.provenance.get("catalog_alleles", [])})
    names = map_allele(cls, alleles, len(obs.mhc))
    names = {
        k.replace("netmhcpan41", "netmhcpan").replace("netmhciipan43", "netmhciipan"): v
        for k, v in (names or {}).items()
    }
    reason = None if key else "unresolved_assay_study_or_mhc"
    if not names:
        reason = reason or "unsupported_mhc_identity"
    result = dict(
        observation_id=obs.observation_id,
        peptide=obs.peptide,
        mhc_class=cls,
        endpoint=obs.endpoint,
        source=obs.source,
        qualifier=obs.qualifier,
        target_log10=target,
        raw_nM=10**target,
        presto=prediction,
        assay_review_status=method.review_status,
        tool_alleles=names,
        exclusion=reason,
    )
    if key:
        endpoint, cls, source, study, prep, mechanism, detector, mhc_key = key
        result.update(
            dict(zip(CELL, (cls, endpoint, source, study, prep, mechanism, detector, mhc_key)))
        )
    return result


@dataclass(frozen=True)
class DTUTool:
    name: str
    executable: str
    version: str
    bundle_home: str

    def __post_init__(self):
        prefix = {"netmhcpan": "4.2", "netmhciipan": "4.3"}.get(self.name)
        if prefix is None or not re.fullmatch(re.escape(prefix) + "[a-z]+", self.version):
            raise ValueError("explicit supported DTU patch version required")


def run_dtu(tool, allele, peptides, output):
    """Run one bounded batch, retaining raw output and verifying the reported version."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    peptides = sorted(set(peptides))
    if not peptides or any(not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", p) for p in peptides):
        raise ValueError("nonempty standard-amino-acid peptide batch required")
    inputs = output / "peptides.txt"
    inputs.write_text("\n".join(peptides) + "\n")
    command = [tool.executable, "-inptype", "1", "-f", str(inputs.resolve()), "-a", allele, "-BA"]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_tool_env(tool.bundle_home),
        timeout=1800,
        check=False,
    )
    (output / "stdout.txt").write_text(proc.stdout)
    (output / "stderr.txt").write_text(proc.stderr)
    receipt = dict(
        tool=asdict(tool),
        argv=command,
        returncode=proc.returncode,
        input_sha256=sha256_file(inputs),
        executable_sha256=sha256_file(tool.executable),
        stdout_sha256=sha256_file(output / "stdout.txt"),
    )
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    display = "NetMHCpan" if tool.name == "netmhcpan" else "NetMHCIIpan"
    if proc.returncode or not re.search(
        rf"{display}\s+version\s+{re.escape(tool.version)}\b", proc.stdout
    ):
        raise RuntimeError("DTU execution failed or reported version differs from pinned version")
    column = "Aff(nM)" if tool.name == "netmhcpan" else "Affinity(nM)"
    values, distance = _table_rows(proc.stdout, column)
    if set(values) != set(peptides) or any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("DTU output missing, extra or nonpositive affinity predictions")
    return [
        dict(
            tool=tool.name,
            version=tool.version,
            allele=allele,
            peptide=p,
            prediction=math.log10(values[p]),
            allele_distance=distance,
        )
        for p in peptides
    ]


def paired_cells(frame, comparator, *, minimum=20):
    """Exact-value metrics on common rows, within complete assay/allele/study cells.

    Caller must record all excluded rows and supply the cohort manifest separately.
    Missing comparator values fail instead of silently changing the population.
    """
    if not set(CELL + ["qualifier", "target_log10", "raw_nM", "presto", comparator]) <= set(frame):
        raise ValueError("incomplete comparison columns")
    if frame[CELL].isna().any().any() or (frame[CELL] == "").any().any():
        raise ValueError("unresolved comparison cell")
    if not np.isfinite(
        frame[["target_log10", "raw_nM", "presto", comparator]].to_numpy(float)
    ).all():
        raise ValueError("comparison requires finite predictions on identical rows")
    if (frame.raw_nM <= 0).any() or not np.allclose(
        np.log10(frame.raw_nM), frame.target_log10, rtol=1e-7, atol=1e-7
    ):
        raise ValueError("physical target columns disagree")
    results = []
    exact = frame[frame.qualifier == "="]
    for key, group in exact.groupby(CELL, sort=True):
        cell = dict(zip(CELL, key))
        eligible = len(group) >= minimum and group.target_log10.nunique() > 1
        for name in ("presto", comparator):
            metrics = _one(
                group.target_log10.to_numpy(float),
                group.raw_nM.to_numpy(float),
                group[name].to_numpy(float),
            )
            results.append(
                dict(**cell, method=name, n=len(group), primary_eligible=eligible, **metrics)
            )
    return pd.DataFrame(results)


def paired_study_interval(cells, comparator, metric, *, draws=2000, seed=20261107):
    """Paired study-cluster uncertainty; positive difference favors Presto for rho.

    The caller supplies one class/endpoint/assay population. Each study averages
    its allele cells equally; study resampling preserves pairing across methods.
    This is descriptive when fewer than three independent studies are available.
    """
    for field in ("mhc_class", "endpoint", "preparation", "mechanism", "detector"):
        if cells[field].nunique() != 1:
            raise ValueError("separate class, endpoint and assay populations before pooling")
    eligible = cells[cells.primary_eligible]
    wide = eligible.pivot(index=CELL, columns="method", values=metric)
    if not {"presto", comparator} <= set(wide) or wide[["presto", comparator]].isna().any().any():
        raise ValueError("paired metric missing or undefined")
    delta = (wide.presto - wide[comparator]).groupby(level=["source", "study"]).mean()
    if not len(delta):
        raise ValueError("no eligible studies")
    rng = np.random.default_rng(seed)
    bootstrap = delta.to_numpy()[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
    return dict(
        metric=metric,
        difference="presto_minus_comparator",
        mean=float(delta.mean()),
        ci_low=float(np.quantile(bootstrap, 0.025)),
        ci_high=float(np.quantile(bootstrap, 0.975)),
        studies=len(delta),
        status="study_cluster_estimate" if len(delta) >= 3 else "descriptive_only",
    )
