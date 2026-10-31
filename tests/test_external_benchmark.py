# ruff: noqa: E501
"""Unit tests for the external-tool benchmark harness (no tool binaries needed)."""

import numpy as np
import pandas as pd
import pytest

from presto.evaluation import external_benchmark as eb

NETMHCPAN_OUT = """
HLA-A0201 : Distance to training data  0.000 (using nearest neighbor HLA-A02:01)

 Pos         MHC        Peptide      Core Of Gp Gl Ip Il        Icore        Identity  Score_EL %Rank_EL Score_BA %Rank_BA  Aff(nM) BindLevel
---------------------------------------------------------------------------------------------------------------------------
   1  HLA-A*0201       SIINFEKL SII-NFEKL  0  0  0  3  1     SIINFEKL         PEPLIST 0.0100620    6.723 0.110414   20.171 15140.42
   1  HLA-A*0201      GILGFVFTL GILGFVFTL  0  0  0  0  0    GILGFVFTL         PEPLIST 0.7899480    0.114 0.816707    0.077     7.27 <= SB
---------------------------------------------------------------------------------------------------------------------------
"""

NETMHCIIPAN_OUT = """
# DRB1_0401 : Distance to training data  0.064 (using nearest neighbor DRB1_0401)
 Pos               MHC              Peptide   Of        Core  Core_Rel Inverted        Identity      Score_EL %Rank_EL  Exp_Bind      Score_BA %Rank_BA  Affinity(nM)  BindLevel
-----
   1         DRB1_0401      PKYVKQNTLKLAT    2   YVKQNTLKL     1.000        0        Sequence      0.715454     1.03        NA      0.545680     3.05        136.41   <= WB
   2         DRB1_0401 MASPGSGFWSFGSEDGSGDS    7   FWSFGSEDG     1.000        0        Sequence      0.172036     7.52        NA      0.242199    52.59       3638.18
"""


def test_parse_tool_outputs():
    values, distance = eb.parse_netmhcpan_output(NETMHCPAN_OUT)
    assert values == {"SIINFEKL": 15140.42, "GILGFVFTL": 7.27}
    assert distance == 0.0
    values, distance = eb.parse_netmhciipan_output(NETMHCIIPAN_OUT)
    assert values == {"PKYVKQNTLKLAT": 136.41, "MASPGSGFWSFGSEDGSGDS": 3638.18}
    assert distance == 0.064


def test_allele_mapping_rules():
    assert eb.map_allele("I", ["HLA-A*02:01"]) == {
        "netmhcpan41": "HLA-A0201",
        "mhcflurry": "HLA-A*02:01",
    }
    assert eb.map_allele("II", ["HLA-DRA*01:01", "HLA-DRB1*04:01"]) == {
        "netmhciipan43": "DRB1_0401"
    }
    assert eb.map_allele("II", ["HLA-DQA1*05:01", "HLA-DQB1*03:03"]) == {
        "netmhciipan43": "HLA-DQA10501-DQB10303"
    }
    # multi-allelic, bag, non-HLA, wrong locus pairing, wrong DRA all excluded
    assert eb.map_allele("I", ["HLA-A*02:01", "HLA-B*07:02"]) is None
    assert eb.map_allele("I", ["HLA-A*02:01"], n_mhc_entries=3) is None
    assert eb.map_allele("I", ["H-2-Kb"]) is None
    assert eb.map_allele("II", ["HLA-DQA1*05:01", "HLA-DRB1*04:01"]) is None
    assert eb.map_allele("II", ["HLA-DRA*01:02", "HLA-DRB1*04:01"]) is None
    assert eb.map_allele("II", []) is None


def _rows():
    base = dict(
        endpoint="ic50", detector="fluorescence", study_id="s1", qualifier="=",
        n_mhc_entries=1, bound_source="explicit", raw_value=100.0, raw_nM=100.0,
        target_log10=2.0, presto=2.0,
    )  # fmt: skip
    return pd.DataFrame(
        [
            dict(base, observation_id="a", peptide="SIINFEKL", mhc_class="I", catalog_alleles="HLA-A*02:01"),
            dict(base, observation_id="b", peptide="SIINFEKL", mhc_class="I", catalog_alleles="HLA-A*02:01|HLA-B*07:02"),
            dict(base, observation_id="c", peptide="SIINFEKX", mhc_class="I", catalog_alleles="HLA-A*02:01"),
            dict(base, observation_id="d", peptide="SIINFEKLSIINFEKLS", mhc_class="I", catalog_alleles="HLA-A*02:01"),
            dict(base, observation_id="e", peptide="SIINFEKL", mhc_class="I", catalog_alleles="HLA-A*02:19"),
            dict(base, observation_id="f", peptide="PKYVKQNTLKLAT", mhc_class="II", catalog_alleles="HLA-DRA*01:01|HLA-DRB1*04:01"),
        ]
    )  # fmt: skip


def test_eligibility_reasons_and_all_method_exclusion():
    supported = {
        "netmhcpan41": ["HLA-A0201"],
        "mhcflurry": ["HLA-A*02:01"],  # A*02:19 unsupported by both
        "netmhciipan43": ["DRB1_0401"],
    }
    out = eb.annotate_eligibility(_rows(), supported)
    reasons = dict(zip(out["observation_id"], out["exclusion"]))
    assert reasons == {
        "a": None,
        "b": "multi_or_ambiguous",
        "c": "bad_residue",
        "d": "length",
        "e": "unsupported_allele",
        "f": None,
    }
    # a row unsupported by only one applicable tool is still dropped for all methods
    only_flurry = dict(supported, mhcflurry=[])
    assert eb.annotate_eligibility(_rows(), only_flurry).loc[0, "exclusion"] == "unsupported_allele"


def test_tool_attachment_and_failure_rule():
    supported = {
        "netmhcpan41": ["HLA-A0201"],
        "mhcflurry": ["HLA-A*02:01"],
        "netmhciipan43": ["DRB1_0401"],
    }
    frame = eb.annotate_eligibility(_rows(), supported)
    preds = pd.DataFrame(
        [
            ("netmhcpan41", "HLA-A0201", "SIINFEKL", 100.0),
            # mhcflurry returns nothing -> tool failure for class I row
            ("netmhciipan43", "DRB1_0401", "PKYVKQNTLKLAT", 1000.0),
        ],
        columns=["tool", "allele", "peptide", "ic50_nM"],
    )
    frame = eb.apply_tool_failures(eb.attach_tool_predictions(frame, preds))
    assert frame.loc[0, "pred_netmhcpan41"] == pytest.approx(2.0)
    assert frame.loc[0, "exclusion"] == "tool_failure"
    assert frame.loc[5, "pred_netmhciipan43"] == pytest.approx(3.0)
    assert frame.loc[5, "exclusion"] is None


def test_subsample_is_deterministic_and_capped():
    frame = pd.DataFrame({"observation_id": [f"o{i}" for i in range(50)], "study_id": "big"})
    first, log = eb.subsample_per_study(frame, cap=10)
    second, _ = eb.subsample_per_study(frame.sample(frac=1, random_state=1), cap=10)
    assert len(first) == 10 and log == [{"study_id": "big", "kept": 10}]
    assert set(first["observation_id"]) == set(second["observation_id"])


def _scored_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for study in ("s1", "s2", "s3"):
        truth = rng.normal(2.5, 1.0, n)
        rows.append(
            pd.DataFrame(
                {
                    "mhc_class": "I", "endpoint": "ic50", "detector": "fluorescence",
                    "study_id": study, "qualifier": "=", "bound_source": "explicit",
                    "target_log10": truth, "raw_nM": 10**truth,
                    "presto": rng.normal(2.5, 1.0, n),  # noise
                    "train_mean": 2.5,
                    "pred_netmhcpan41": truth + rng.normal(0, 0.2, n),  # near-perfect
                    "pred_mhcflurry": truth + rng.normal(0, 0.5, n),
                }
            )
        )  # fmt: skip
    return pd.concat(rows, ignore_index=True)


def test_unit_and_pooled_metrics_rank_the_methods():
    frame = _scored_frame()
    units = eb.unit_metrics(frame, row_draws=50)
    assert set(units["method"]) == {"presto", "train_mean", "netmhcpan41", "mhcflurry"}
    sp = units.pivot(index="study_id", columns="method", values="spearman")
    assert (sp["netmhcpan41"] > 0.9).all() and (sp["presto"].abs() < 0.5).all()
    assert sp["train_mean"].isna().all()  # constant predictor: Spearman undefined
    assert units["rmse"].notna().all()
    pooled = eb.pooled_metrics(units, draws=200)
    row = pooled[(pooled["method"] == "netmhcpan41") & (pooled["metric"] == "spearman")].iloc[0]
    assert row["n_studies"] == 3 and row["diff_vs_presto"] > 0.5
    assert row["fisher_z_weighted"] > 0.9
    # pooling never mixes detectors
    assert pooled[["mhc_class", "detector"]].drop_duplicates().shape[0] == 1


def test_small_units_are_excluded_from_pooling():
    frame = _scored_frame(n=10)
    units = eb.unit_metrics(frame, row_draws=0)
    assert eb.pooled_metrics(units, draws=10).empty


def test_censored_bound_satisfaction():
    frame = pd.DataFrame(
        {
            "mhc_class": "I", "endpoint": "ic50", "detector": "fluorescence",
            "study_id": "s", "qualifier": [">", ">", ">="], "bound_source": "inferred",
            "raw_nM": [1000.0, 1000.0, 10000.0], "presto": [2.0, 4.0, 3.0],
            "train_mean": 2.5, "pred_netmhcpan41": 5.0, "pred_mhcflurry": 5.0,
        }
    )  # fmt: skip
    table = eb.censored_metrics(frame)
    presto = table[table["method"] == "presto"].iloc[0]
    assert presto["n"] == 3 and presto["satisfied"] == pytest.approx(1 / 3)
    assert presto["median_violation_log10"] == pytest.approx(1.0)
    tool = table[table["method"] == "netmhcpan41"].iloc[0]
    assert tool["satisfied"] == 1.0


def test_qualitative_auroc_uses_identical_rows():
    frame = pd.DataFrame(
        {
            "mhc_class": "I", "study_id": "q", "raw_value": [1, 1, 0, 0],
            "presto": [0.9, 0.8, 0.2, 0.1], "pred_netmhcpan41": [1.0, 2.0, 4.0, 5.0],
            "pred_mhcflurry": [5.0, 4.0, 2.0, 1.0],
        }
    )  # fmt: skip
    table = eb.qualitative_metrics(frame).set_index("method")
    assert table.loc["presto", "auroc"] == 1.0
    assert table.loc["netmhcpan41", "auroc"] == 1.0
    assert table.loc["mhcflurry", "auroc"] == 0.0
    assert (table["n"] == 4).all()


def test_census_counts_every_row_once():
    frame = _rows()
    frame["exclusion"] = [None, "length", None, "bad_residue", None, None]
    census = eb.exclusion_census(frame)
    assert census["n"].sum() == len(frame)


def test_detector_falls_back_to_raw_assay_method():
    row = {
        "endpoint": "ic50", "observation_id": "x", "peptide": "SIINFEKL", "mhc_classes": ["I"],
        "qualifier": "=", "raw_value": 5.0, "raw_unit": "nM", "target": 0.7, "prediction": 0.6,
        "observation": {
            "mhc": [{"provenance": {"catalog_alleles": ["HLA-A*02:01"]}}],
            "provenance": {
                "study_id": "s",
                "raw_record": {"assay_method": "purified MHC/competitive/radioactivity"},
            },
        },
    }  # fmt: skip
    assert eb.extract_record(row)["detector"] == "radioactivity"
    row["observation"]["assay_method"] = {"detector": "fluorescence"}
    assert eb.extract_record(row)["detector"] == "fluorescence"
