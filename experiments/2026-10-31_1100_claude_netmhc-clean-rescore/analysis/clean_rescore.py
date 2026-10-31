"""Tool-clean re-scoring of the external NetMHC benchmark (pre-registered plan:
experiments/agents/claude/plans/2026-10-31_netmhc-clean-rescore.md).

Usage (python -I not usable because presto must import; tool data is only READ as text):
  python clean_rescore.py --dumps-dir D --pan41 DIR --ii43 DIR --mhcflurry-train FILE
      --mhcflurry-affinity FILE --pr-results DIR --out-dir EXP --cache-dir C

Tool training files are parsed as plain text and never executed or committed.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import glob
import hashlib
import itertools
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from presto.evaluation import external_benchmark as eb
from presto.training.holdout_eval import spearman

LABELS = [
    "direct-s42_test", "direct-s43_test", "curated-multi-s42_test", "curated-multi-s43_test",
    "direct-s42_validation", "direct-s43_validation",
    "curated-multi-s42_validation", "curated-multi-s43_validation",
]  # fmt: skip
TOOL_CLASS = {"netmhcpan41": "I", "mhcflurry": "I", "netmhciipan43": "II"}
TOOL_COL = {"netmhcpan41": "allele_netmhcpan41", "mhcflurry": "allele_mhcflurry",
            "netmhciipan43": "allele_netmhciipan43"}  # fmt: skip
KMER = {"I": 8, "II": 9}
VARIANTS = ("pair", "peptide", "nested")
GAP_METRICS = ("spearman", "pearson", "rmse", "auroc")


def norm(a: str) -> str:
    return a.replace("HLA-", "").replace("*", "").replace(":", "").upper()


def key_hash(peptide: str, allele: str) -> str:
    return hashlib.sha256(f"{peptide}|{allele}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# training-data streams: yield (peptide, tuple_of_normalised_alleles)
# ---------------------------------------------------------------------------
def _allelelist(path):
    out = {}
    for line in open(path):
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = tuple(sorted({norm(x) for x in parts[1].strip().split(",") if x}))
    return out


def stream_dtu(root: str):
    """BA files (col0 peptide, col2 allele) and EL label-1 lines via allelelist."""
    alist = _allelelist(os.path.join(root, "allelelist"))
    stats = {"ba": 0, "el_pos": 0, "el_unmapped_cell": 0}
    for f in sorted(glob.glob(os.path.join(root, "c00?_ba"))):
        for line in open(f):
            p = line.split()
            if len(p) >= 3:
                stats["ba"] += 1
                yield p[0], (norm(p[2]),)
    for f in sorted(glob.glob(os.path.join(root, "c00?_el"))):
        for line in open(f):
            p = line.split()
            if len(p) >= 3 and p[1] == "1":
                alleles = alist.get(p[2])
                if alleles is None:
                    stats["el_unmapped_cell"] += 1
                    continue
                stats["el_pos"] += 1
                yield p[0], alleles
    print("stream stats", root, stats, flush=True)


def stream_mhcflurry(path: str):
    for r in csv.DictReader(bz2.open(path, "rt")):
        yield r["peptide"], (norm(r["allele"]),)


def mark_seen(stream, tests: pd.DataFrame, cls: str) -> pd.DataFrame:
    """tests: columns peptide, nallele. Returns bool columns pair/peptide/nested."""
    k = KMER[cls]
    pair_keys = defaultdict(list)
    pep_idx = defaultdict(list)
    kmer_idx = defaultdict(list)
    for i, (p, a) in enumerate(zip(tests["peptide"], tests["nallele"])):
        pair_keys[(a, p)].append(i)
        pep_idx[p].append(i)
        for j in range(len(p) - k + 1):
            kmer_idx[p[j : j + k]].append(i)
    n = len(tests)
    out = {v: np.zeros(n, bool) for v in VARIANTS}
    nallele = tests["nallele"].to_numpy()
    peps = tests["peptide"].to_numpy()
    for tp, alleles in stream:
        hit_pep = pep_idx.get(tp)
        if hit_pep:
            for i in hit_pep:
                out["peptide"][i] = True
        for a in alleles:
            h = pair_keys.get((a, tp))
            if h:
                for i in h:
                    out["pair"][i] = True
        if len(tp) >= k:
            cand = set()
            for j in range(len(tp) - k + 1):
                c = kmer_idx.get(tp[j : j + k])
                if c:
                    cand.update(c)
            for i in cand:
                if out["nested"][i] or nallele[i] not in alleles:
                    continue
                if tp in peps[i] or peps[i] in tp:
                    out["nested"][i] = True
    out["nested"] |= out["pair"]
    out["peptide"] |= out["pair"]
    return pd.DataFrame(out, index=tests.index)


def build_seen(frames: dict, args) -> pd.DataFrame:
    cache = Path(args.cache_dir) / "seen_flags.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    allrows = pd.concat(
        [f.assign(label=l) for l, f in frames.items()], ignore_index=True
    )
    uniq = allrows.drop_duplicates(["observation_id", "peptide"]).copy()
    sources = {
        "netmhcpan41": lambda: stream_dtu(args.pan41),
        "netmhciipan43": lambda: stream_dtu(args.ii43),
        "mhcflurry": lambda: itertools.chain(
            stream_mhcflurry(args.mhcflurry_train), stream_mhcflurry(args.mhcflurry_affinity)
        ),
        "mhcflurry_affonly": lambda: stream_mhcflurry(args.mhcflurry_affinity),
    }
    result = uniq[["observation_id", "peptide", "mhc_class"]].copy()
    for tool, make in sources.items():
        base = tool.replace("_affonly", "")
        cls = TOOL_CLASS[base]
        sub = uniq[(uniq["mhc_class"] == cls) & uniq[TOOL_COL[base]].notna()]
        tests = pd.DataFrame(
            {"peptide": sub["peptide"], "nallele": sub[TOOL_COL[base]].map(norm)}, index=sub.index
        )
        res = mark_seen(make(), tests, cls)
        for v in VARIANTS:
            col = pd.Series(np.nan, index=uniq.index, dtype=object)
            col.loc[res.index] = res[v]
            result[f"seen_{v}_{tool}"] = col
        print(tool, "tests", len(tests), {v: int(res[v].sum()) for v in VARIANTS}, flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    return result


# ---------------------------------------------------------------------------
def tool_applicable(cls):
    return eb.CLASS_TOOLS[cls]


def seen_cols(frame: pd.DataFrame, variant: str, tool: str) -> pd.Series:
    return frame[f"seen_{variant}_{tool}"].astype("boolean")


def subset_mask(frame, variant, scope):
    """True = row in the subset. scope = tool name or 'common'."""
    keep = pd.Series(True, index=frame.index)
    for cls in ("I", "II"):
        isc = frame["mhc_class"] == cls
        tools = tool_applicable(cls) if scope == "common" else (scope,)
        if scope != "common" and TOOL_CLASS[scope] != cls:
            keep &= ~isc  # tool not applicable to this class: no rows
            continue
        for t in tools:
            s = seen_cols(frame, variant, t)
            keep &= ~(isc & s.fillna(True).astype(bool))
    return keep


def pool_all(exact_subset: pd.DataFrame):
    units = eb.unit_metrics(exact_subset, row_draws=0)
    pooled = eb.pooled_metrics(units, draws=2000)
    return units, pooled


def unit_key_tuples(units, cols):
    ok = units[units["n"] >= eb.MIN_UNIT_ROWS]
    return set(map(tuple, ok[cols].drop_duplicates().to_numpy()))


def seen_vs_unseen(exact: pd.DataFrame, variant: str, draws=2000):
    rows = []
    for key, g in exact.groupby(eb.UNIT_KEYS, sort=True):
        cls = key[0]
        for tool in tool_applicable(cls):
            s = seen_cols(g, variant, tool).fillna(False).astype(bool).to_numpy()
            if s.sum() < eb.MIN_UNIT_ROWS or (~s).sum() < eb.MIN_UNIT_ROWS:
                continue
            truth = g["target_log10"].to_numpy(float)
            rec = dict(zip(eb.UNIT_KEYS, key), tool=tool, variant=variant,
                       n_seen=int(s.sum()), n_unseen=int((~s).sum()))  # fmt: skip
            for m, pred in (("tool", g[f"pred_{tool}"].to_numpy(float)),
                            ("presto", g["presto"].to_numpy(float))):  # fmt: skip
                a, b = spearman(truth[s], pred[s]), spearman(truth[~s], pred[~s])
                rec[f"{m}_spearman_seen"] = np.nan if a is None else a
                rec[f"{m}_spearman_unseen"] = np.nan if b is None else b
            rows.append(rec)
    units = pd.DataFrame(rows)
    pooled = []
    if len(units):
        rng = np.random.default_rng(eb.BOOTSTRAP_SEED)
        for (cls, det, tool), g in units.groupby(["mhc_class", "detector", "tool"]):
            per = g.groupby("study_id")[[c for c in g.columns if "spearman" in c]].mean()
            rec = dict(mhc_class=cls, detector=det, tool=tool, variant=variant,
                       n_studies=len(per))  # fmt: skip
            for m in ("tool", "presto"):
                d = (per[f"{m}_spearman_unseen"] - per[f"{m}_spearman_seen"]).to_numpy(float)
                d = d[np.isfinite(d)]
                rec[f"{m}_unseen_minus_seen"] = float(d.mean()) if len(d) else np.nan
                if len(d):
                    idx = rng.integers(0, len(d), size=(draws, len(d)))
                    rec[f"{m}_ci_lo"], rec[f"{m}_ci_hi"] = eb._ci(d[idx].mean(axis=1))
                rec[f"{m}_mean_seen"] = float(per[f"{m}_spearman_seen"].mean())
                rec[f"{m}_mean_unseen"] = float(per[f"{m}_spearman_unseen"].mean())
            pooled.append(rec)
    return units, pd.DataFrame(pooled)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps-dir", required=True)
    ap.add_argument("--pan41", required=True)
    ap.add_argument("--ii43", required=True)
    ap.add_argument("--mhcflurry-train", required=True)
    ap.add_argument("--mhcflurry-affinity", required=True)
    ap.add_argument("--pr-results", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)

    frames = {
        l: pd.read_csv(Path(args.dumps_dir) / f"predictions_{l}.csv.gz") for l in LABELS
    }
    seen = build_seen(frames, args)
    seen_idx = seen.drop_duplicates(["observation_id", "peptide"]).set_index(
        ["observation_id", "peptide"]
    )
    flag_cols = [c for c in seen.columns if c.startswith("seen_")]

    # ---- mask (hash keyed) ----
    mask_parts = []
    for l, f in frames.items():
        split = l.split("_")[-1]
        m = f[["observation_id", "peptide", "mhc_class", "allele_mhcflurry",
               "allele_netmhciipan43"]].copy()  # fmt: skip
        m["allele_key"] = np.where(m["mhc_class"] == "I", m["allele_mhcflurry"],
                                   m["allele_netmhciipan43"])  # fmt: skip
        m["peptide_allele_sha256"] = [key_hash(p, a) for p, a in zip(m["peptide"], m["allele_key"])]
        m = m.join(seen_idx[flag_cols], on=["observation_id", "peptide"])
        m["split"] = split
        mask_parts.append(m[["split", "observation_id", "mhc_class", "peptide_allele_sha256",
                             *flag_cols]])  # fmt: skip
    mask = pd.concat(mask_parts).drop_duplicates(["split", "observation_id"])
    mask.drop(columns=["observation_id"]).to_csv(
        out / "presto_row_overlap_mask.csv.gz", index=False
    )

    scopes = ["netmhcpan41", "mhcflurry", "netmhciipan43", "common"]
    all_units, all_pooled, fracs, svu_units, svu_pooled = [], [], [], [], []
    for label, frame in frames.items():
        frame = frame.join(seen_idx[flag_cols], on=["observation_id", "peptide"])
        eligible = frame[frame["exclusion"].isna()]
        ba = eligible[eligible["endpoint"].isin(eb.BA_ENDPOINTS)]
        exact = ba[ba["qualifier"] == "="]
        # seen fractions
        for cls in ("I", "II"):
            for tool in tool_applicable(cls):
                for v in VARIANTS:
                    for scope_name, d in (("exact_ba", exact), ("all_eligible", eligible)):
                        d = d[d["mhc_class"] == cls]
                        s = seen_cols(d, v, tool).fillna(False).astype(bool)
                        t = d.assign(seen=s).groupby(["detector", "study_id"]).agg(
                            n=("seen", "size"), n_seen=("seen", "sum")).reset_index()
                        t["frac_seen"] = t["n_seen"] / t["n"]
                        t.insert(0, "label", label); t.insert(1, "mhc_class", cls)
                        t.insert(2, "tool", tool); t.insert(3, "variant", v)
                        t.insert(4, "rows", scope_name)  # fmt: skip
                        fracs.append(t)
        # full + clean subsets
        subsets = {"full": exact}
        for v in VARIANTS:
            for scope in scopes:
                subsets[f"clean_{scope}__{v}"] = exact[subset_mask(exact, v, scope)]
        full_units, full_pooled = pool_all(exact)
        cols = ["mhc_class", "endpoint", "detector", "study_id"]
        results = {"full": (full_units, full_pooled)}
        for name, sub in subsets.items():
            if name == "full" or sub.empty:
                continue
            u, p = pool_all(sub)
            results[name] = (u, p)
            if len(u):
                keys = unit_key_tuples(u, cols)
                mk = full_units[[tuple(r) in keys for r in full_units[cols].to_numpy()]]
                results[f"fullmatch_{name}"] = (mk, eb.pooled_metrics(mk, draws=2000))
        for name, (u, p) in results.items():
            all_units.append(u.assign(label=label, subset=name))
            all_pooled.append(p.assign(label=label, subset=name))
        for v in VARIANTS:
            su, sp = seen_vs_unseen(exact, v)
            svu_units.append(su.assign(label=label))
            svu_pooled.append(sp.assign(label=label))
        print("done", label, flush=True)

    units = pd.concat(all_units, ignore_index=True)
    pooled = pd.concat(all_pooled, ignore_index=True)
    units.to_csv(out / "units_all.csv.gz", index=False)
    pooled.to_csv(out / "pooled_all.csv", index=False)
    pd.concat(fracs).to_csv(out / "seen_fraction_by_study.csv", index=False)
    pd.concat(svu_units).to_csv(out / "seen_vs_unseen_units.csv", index=False)
    pd.concat(svu_pooled).to_csv(out / "seen_vs_unseen_pooled.csv", index=False)

    # ---- reproduction check against PR #149 ----
    chk = []
    for l in LABELS:
        old = pd.read_csv(Path(args.pr_results) / f"{l}__pooled.csv")
        new = pooled[(pooled["label"] == l) & (pooled["subset"] == "full")]
        key = ["mhc_class", "detector", "method", "metric"]
        mg = old.merge(new, on=key, suffixes=("_old", "_new"))
        d = (mg["mean_of_study_old"] - mg["mean_of_study_new"]).abs().max()
        chk.append(dict(label=l, cells=len(mg), n_old=len(old), n_new=len(new), max_abs_diff=d))
    pd.DataFrame(chk).to_csv(out / "reproduction_check.csv", index=False)
    print(pd.DataFrame(chk))

    # ---- gap table before / after ----
    g = []
    def cell(label, subset, tool, cls, det, metric):
        p = pooled[(pooled["label"] == label) & (pooled["subset"] == subset)
                   & (pooled["mhc_class"] == cls) & (pooled["detector"] == det)
                   & (pooled["metric"] == metric)]  # fmt: skip
        t = p[p["method"] == tool]
        pr = p[p["method"] == "presto"]
        if t.empty or pr.empty:
            return None
        t, pr = t.iloc[0], pr.iloc[0]
        return dict(n_studies=int(t["n_studies"]), n_rows=int(t["n_rows"]),
                    tool=t["mean_of_study"], presto=pr["mean_of_study"],
                    gap=t["diff_vs_presto"], gap_lo=t["diff_ci_lo"], gap_hi=t["diff_ci_hi"])  # fmt: skip
    for label in LABELS:
        for v in VARIANTS:
            for scope in scopes:
                for tool in ([scope] if scope != "common" else list(eb.ALL_TOOLS)):
                    cls = TOOL_CLASS[tool]
                    name = f"clean_{scope}__{v}"
                    dets = sorted(pooled[(pooled["label"] == label)
                                         & (pooled["mhc_class"] == cls)]["detector"].unique())  # fmt: skip
                    for det in dets:
                        for metric in GAP_METRICS:
                            rec = dict(label=label, variant=v, scope=scope, tool=tool,
                                       mhc_class=cls, detector=det, metric=metric)  # fmt: skip
                            for tag, sname in (("full", "full"), ("fullmatch", f"fullmatch_{name}"),
                                               ("clean", name)):  # fmt: skip
                                c = cell(label, sname, tool, cls, det, metric)
                                if c:
                                    rec.update({f"{tag}_{k}": val for k, val in c.items()})
                            if "clean_n_studies" in rec:
                                s = rec["clean_n_studies"]
                                rec["reportable"] = ("headline" if s >= 3
                                                     else "thin(<3 studies)" if s >= 1 else "no")  # fmt: skip
                            else:
                                rec["reportable"] = "no(clean n<20 everywhere)"
                            g.append(rec)
    gap = pd.DataFrame(g)
    gap.to_csv(out / "gap_before_after.csv", index=False)
    (out / "dump_sha256.json").write_text(json.dumps(
        {l: eb.sha256_file(Path(args.dumps_dir) / f"predictions_{l}.csv.gz") for l in LABELS},
        indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
