"""Tests for candidate junction modes ('masked', 'sample', 'mil')."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from presto.data.hitlist_binding_expansion import ProtectedIdentityIndex
from presto.data.hitlist_nonhuman_ms_expansion import build_hitlist_nonhuman_ms_expansion
from presto.data.modular import Context, MHCInput, Observation, SourceJunction, TargetScale
from presto.models.presto import Presto, PrestoConfig
from presto.training.modular import load_checkpoint, save_checkpoint, train_step

G1_A = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
G2_A = "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"


def _make_config(mode: str = "masked", temp: float = 1.0) -> PrestoConfig:
    return PrestoConfig(
        endpoints=("ic50", "elution_observation"),
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_shared_processing=True,
        shared_processing_v2=True,
        elution_pmhc_residual=True,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        query_anchor_embedding=True,
        groove_pocket_pool=True,
        register_pooling_mode="thermodynamic",
        candidate_junction_mode=mode,
        candidate_junction_temp=temp,
    )


def _elution_obs(
    obs_id: str,
    *,
    source_n: str | None = None,
    source_c: str | None = None,
    n_terminus: bool | None = None,
    c_terminus: bool | None = None,
    candidate_junctions: tuple[SourceJunction, ...] = (),
    endpoint: str = "elution_observation",
) -> Observation:
    return Observation(
        observation_id=obs_id,
        source="test_src",
        endpoint=endpoint,
        peptide="SIINFEKLA",
        mhc=(MHCInput(groove1=G1_A, groove2=G2_A, mhc_class="I"),),
        value=1.0 if endpoint == "elution_observation" else 50.0,
        unit="binary" if endpoint == "elution_observation" else "nM",
        split="train",
        group_id="g1",
        context=Context(
            source_n=source_n,
            source_c=source_c,
            n_terminus=n_terminus,
            c_terminus=c_terminus,
            candidate_junctions=candidate_junctions,
        ),
    )


def test_serialization_and_config_backward_compatibility(tmp_path: Path):
    # Empty candidate_junctions omitted from Observation.to_dict()
    obs_plain = _elution_obs("plain", source_n="ACDEF", source_c="GHIKL")
    d_plain = obs_plain.to_dict()
    assert "candidate_junctions" not in d_plain["context"]
    assert Observation.from_dict(d_plain) == obs_plain

    # Non-empty candidate_junctions round-trips cleanly with weights
    j1 = SourceJunction("AAAAACDEF", "GHIKLMMMM", False, False, weight=3.0)
    j2 = SourceJunction("DDDDDCDEF", "GHIKLPPPP", False, False, weight=1.0)
    obs_cj = _elution_obs("cj", candidate_junctions=(j1, j2))
    obs_cj.validate()
    d_cj = obs_cj.to_dict()
    assert len(d_cj["context"]["candidate_junctions"]) == 2
    assert Observation.from_dict(d_cj) == obs_cj

    # Invalid SourceJunction weight rejected
    with pytest.raises(ValueError, match="weight"):
        SourceJunction("ACDEF", "GHIKL", False, False, weight=0.0).validate()

    # Default masked config omits new keys; zero new parameters across modes
    cfg_masked = _make_config("masked")
    assert "candidate_junction_mode" not in cfg_masked.to_dict()
    assert "candidate_junction_temp" not in cfg_masked.to_dict()

    cfg_mil = _make_config("mil", temp=0.8)
    assert cfg_mil.to_dict()["candidate_junction_mode"] == "mil"
    assert cfg_mil.to_dict()["candidate_junction_temp"] == 0.8

    torch.manual_seed(7)
    m_masked = Presto(cfg_masked)
    m_mil = Presto(cfg_mil)
    assert set(m_masked.state_dict()) == set(m_mil.state_dict())

    # Save/reload preserves candidate_junction_mode
    ckpt = tmp_path / "mil.pt"
    save_checkpoint(ckpt, m_mil, {"ic50": TargetScale(mean=2.5, scale=1.0, exact_count=10)})
    reloaded, _, _ = load_checkpoint(ckpt, device="cpu")
    assert reloaded.config.candidate_junction_mode == "mil"
    assert reloaded.config.candidate_junction_temp == 0.8


def test_mil_zero_bag_size_bias_and_prior_weighting():
    torch.manual_seed(11)
    model = Presto(_make_config("mil", temp=1.0))
    # Perturb zero-init readout weights slightly so flanks have non-trivial distinct scores
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "elution_shared" in name and p.ndim >= 1:
                p.add_(0.25 * torch.randn_like(p))
    model.eval()

    j_a = SourceJunction("AAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCC", False, False)
    j_b = SourceJunction("WWWWWWWWWWWWWWW", "YYYYYYYYYYYYYYY", False, False)

    # 1. Exact 0.0 log(M) bag-size bias: M=1 vs M=2 vs M=5 identical copies of j_a
    r1 = _elution_obs(
        "r1", source_n=j_a.source_n, source_c=j_a.source_c, n_terminus=False, c_terminus=False
    )
    r2 = _elution_obs("r2", candidate_junctions=(j_a, j_a))
    r5 = _elution_obs("r5", candidate_junctions=(j_a, j_a, j_a, j_a, j_a))
    with torch.no_grad():
        o1, o2, o5 = model([r1, r2, r5])
    assert abs(float(o1.value) - float(o2.value)) < 1e-5
    assert abs(float(o1.value) - float(o5.value)) < 1e-5

    # 2. Prior RNA-seq weight shifts MIL prediction smoothly toward higher-weighted paralog
    r_b = _elution_obs(
        "rb", source_n=j_b.source_n, source_c=j_b.source_c, n_terminus=False, c_terminus=False
    )
    r_unif = _elution_obs("ru", candidate_junctions=(j_a, j_b))
    r_heavy_a = _elution_obs(
        "rha",
        candidate_junctions=(
            SourceJunction(j_a.source_n, j_a.source_c, False, False, weight=1000.0),
            SourceJunction(j_b.source_n, j_b.source_c, False, False, weight=1.0),
        ),
    )
    r_heavy_b = _elution_obs(
        "rhb",
        candidate_junctions=(
            SourceJunction(j_a.source_n, j_a.source_c, False, False, weight=1.0),
            SourceJunction(j_b.source_n, j_b.source_c, False, False, weight=1000.0),
        ),
    )
    with torch.no_grad():
        ob, ou, oha, ohb = model([r_b, r_unif, r_heavy_a, r_heavy_b])
    assert abs(float(oha.value) - float(o1.value)) < 1e-2
    assert abs(float(ohb.value) - float(ob.value)) < 1e-2
    w_unif = ou.diagnostics["candidates"][0]["registers"][0]["elution_terms"][
        "candidate_junction_weights"
    ]
    assert len(w_unif) == 2 and pytest.approx(sum(w_unif), rel=1e-5) == 1.0
    w_ha = oha.diagnostics["candidates"][0]["registers"][0]["elution_terms"][
        "candidate_junction_weights"
    ]
    assert w_ha[0] > 0.95


def test_sample_mode_stochastic_train_deterministic_eval_and_affinity_invariance():
    torch.manual_seed(17)
    j_a = SourceJunction("AAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCC", False, False)
    j_b = SourceJunction("WWWWWWWWWWWWWWW", "YYYYYYYYYYYYYYY", False, False)

    for mode in ("masked", "sample", "mil"):
        m = Presto(_make_config(mode))
        m.eval()
        q_clean = _elution_obs("q0", endpoint="ic50")
        q_cands = _elution_obs("q1", candidate_junctions=(j_a, j_b), endpoint="ic50")
        with torch.no_grad():
            o_clean, o_cands = m([q_clean, q_cands])
        # Exact 0.0 affinity context invariance across all 3 modes!
        assert float(o_clean.value) == float(o_cands.value)

    # Sample mode: deterministic eval + respects weights in train
    m_sample = Presto(_make_config("sample"))
    with torch.no_grad():
        for name, p in m_sample.named_parameters():
            if "elution_shared" in name and p.ndim >= 1:
                p.add_(0.25 * torch.randn_like(p))
    r_cands = _elution_obs("rc", candidate_junctions=(j_a, j_b))
    m_sample.eval()
    with torch.no_grad():
        ev1 = float(m_sample([r_cands])[0].value)
        ev2 = float(m_sample([r_cands])[0].value)
    assert ev1 == ev2

    # In train mode with heavy prior on j_a, sample matches j_a directly
    r_a = _elution_obs(
        "ra", source_n=j_a.source_n, source_c=j_a.source_c, n_terminus=False, c_terminus=False
    )
    r_bias_a = _elution_obs(
        "rba",
        candidate_junctions=(
            SourceJunction(j_a.source_n, j_a.source_c, False, False, weight=1e9),
            SourceJunction(j_b.source_n, j_b.source_c, False, False, weight=1e-9),
        ),
    )
    m_sample.train()
    with torch.no_grad():
        oa = float(m_sample([r_a])[0].value)
        oba = float(m_sample([r_bias_a])[0].value)
    assert abs(oa - oba) < 1e-5


def test_nonhuman_ms_builder_symmetric_candidate_junctions_all_three_modes(
    tmp_path: Path, monkeypatch
):
    import presto.data.hitlist_nonhuman_ms_expansion as mod

    class _FakeResolver:
        def __init__(self, mhc_full_seqs_csv=None, **kwargs):
            pass

        def resolve(self, restriction: str, mhc_class: str):
            return (G1_A, G2_A) if mhc_class == "I" else None

    monkeypatch.setattr(mod, "FastGrooveResolver", _FakeResolver)

    # Cross-gene paralog peptide VVVWWWYYY appears in 2 distinct genes with different flanks
    mouse_fa = tmp_path / "mouse.fa"
    mouse_fa.write_text(
        ">ENSMUSP002 gene:ENSMUSG002\n"
        "AAAAAAAAAAAAAAAVVVWWWYYYCCCCCCCCCCCCCCCMNPQRSTVWYDEFGHIKLMNPQRSTVWYDEFGHIK\n"
        ">ENSMUSP003 gene:ENSMUSG003\n"
        "DDDDDDDDDDDDDDDVVVWWWYYYEEEEEEEEEEEEEEEQRVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL\n",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        [
            {
                "peptide": "VVVWWWYYY",
                "mhc_species": "Mus musculus",
                "mhc_class": "I",
                "mhc_restriction": "H2-K*b",
                "mhc_allele_set": "",
                "has_ptm": False,
                "is_non_peptide_ligand": False,
                "is_potential_contaminant": None,
                "restriction_evidence": "unknown",
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
                "pmid": 12345,
                "assay_iri": "iri:1",
                "reference_iri": "ref:1",
            }
        ]
    )
    obs_pq = tmp_path / "obs.parquet"
    df.to_parquet(obs_pq, index=False)
    out_gz = tmp_path / "nh.jsonl.gz"

    stats = build_hitlist_nonhuman_ms_expansion(
        obs_pq,
        ProtectedIdentityIndex(set()),
        proteome_paths={"mus_musculus": mouse_fa},
        out_path=out_gz,
        max_candidate_junctions=5,
        neg_per_pos=1,
    )
    assert stats["counts"]["ambiguous_with_candidate_junctions"] == 1
    with gzip.open(out_gz, "rt", encoding="utf-8") as f:
        rows = [Observation.from_dict(json.loads(line)) for line in f]
    assert len(rows) == 2
    pos_r = next(r for r in rows if r.value == 1.0)
    neg_r = next(r for r in rows if r.value == 0.0)
    # Default masked fields stay None symmetrically for "masked" mode
    assert pos_r.context.source_n is None and neg_r.context.source_n is None
    # Candidate junctions populated symmetrically for "sample" and "mil" modes
    assert len(pos_r.context.candidate_junctions) == 2
    assert len(neg_r.context.candidate_junctions) == 1
    assert neg_r.context.candidate_junctions[0].source_n is not None

    # Train step succeeds cleanly in all 3 modes!
    for mode in ("masked", "sample", "mil"):
        model = Presto(_make_config(mode))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        res = train_step(model, opt, rows, {})
        assert torch.isfinite(torch.tensor(res["loss"])).item()
