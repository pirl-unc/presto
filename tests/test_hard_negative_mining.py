"""Unit and integration tests for multi-negative affinity-matched hard-pair mining."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from presto.data.affinity_matching import candidate_key
from presto.data.hard_negative_mining import (
    MiningConfig,
    PrestoAffinityScorer,
    mine_affinity_matched_backgrounds,
)
from presto.data.hitlist_binding_expansion import ProtectedIdentityIndex
from presto.data.modular import Context, Exposure, MHCInput, Observation, TargetScale
from presto.data.presentation import Protein, junction
from presto.models.presto import Presto, PrestoConfig
from presto.training.modular import train_step
from presto.training.processing_pairs import PAIR_KEY, PairPool, pair_loss, validate_pair

G1_A = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
G2_A = "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
G1_B = "GPHSMRYFETSVSRPGLGEPRFIIVGYVDDTQFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGNEQSFRVDLRTLLGYYNQSKG"
G2_B = "GSHTIQVISGCEVGSDGRLLRGYQQYAYDGCDYIALNEDLKTWTAADMAALITKHKWEQAGEAERLRAYLEGTCVEWLRRYLKNGNATLL"
G1_C2 = "EDDIEADHVGTYGISVYQSPGDIGQYTFEFDGDELFYVDLDKKETVWMLPEFGQLASFDPQGGLQNIAVVKHNLGVLTKRSNSTPATNE"
G2_C2 = (
    "GDSERHFVYQFMGECYFTNGTQRIRYVTRYIYNREEYVRYDSDVGEHRAVTELGRPDAEYWNSQPEILERTRAELDTVCRHNYEGPETHTS"
)


class _DictScorer:
    """Deterministic table scorer returning log10(IC50 nM) by (peptide, candidate_key)."""

    def __init__(self, table: dict[tuple[str, str], float], default: float = 4.5):
        self.table = table
        self.default = default

    def score_pairs(self, pairs):
        return [self.table.get((pep, candidate_key(mhc)), self.default) for pep, mhc in pairs]


def _make_pos(
    obs_id: str,
    pep: str,
    mhc: tuple[MHCInput, ...],
    protein_id: str,
    seq: str,
    *,
    mask_flanks: bool = False,
    cell_type: str | None = None,
    cytokines: dict[str, Exposure] | None = None,
    study_id: str = "study_1",
) -> Observation:
    start = seq.index(pep)
    junc = (
        {"source_n": None, "source_c": None, "n_terminus": None, "c_terminus": None}
        if mask_flanks
        else junction(seq, start, len(pep), flank_length=15)
    )
    return Observation(
        observation_id=obs_id,
        source="hitlist_native_ms",
        endpoint="elution_observation",
        peptide=pep,
        mhc=mhc,
        value=1.0,
        unit="binary",
        qualifier="=",
        split="train",
        group_id=f"fam:{protein_id}",
        context=Context(
            cell_type=cell_type,
            cytokines=cytokines or {},
            **junc,
        ),
        provenance={"source_protein": protein_id, "study_id": study_id},
    )


def test_multi_negative_diverse_mining_and_overlap_exclusion(tmp_path: Path):
    mhc_i = MHCInput(groove1=G1_A, groove2=G2_A, mhc_class="I")
    k_i = candidate_key(mhc_i)

    # Construct a canonical 20-AA protein with:
    # - pos peptide PPSPEPTID at offset 15
    # - 3 distinct well-spaced caliper-matching binders (NEGAANAAA, NEGAWNFFF, NEGTHRCCC)
    # - an embargoed binder EMBARGDDD (canonical: EMMARGDDD) and MS-observed binder KKSPEPEEE
    seq = (
        "ACDEFGHIKLMNPQR"  # 0..14
        "PPSPEPTID"  # 15..23
        "VWYACDEFGHIKLMN"  # 24..38
        "NEGAANAAA"  # 39..47 (matches pos)
        "PQRSTVWYACDEFGH"  # 48..62
        "NEGAWNFFF"  # 63..71 (matches pos)
        "IKLMNPQRSTVWYAC"  # 72..86
        "EMMARGDDD"  # 87..95 (embargoed!)
        "DEFGHIKLMNPQRST"  # 96..110
        "KKSPEPEEE"  # 111..119 (globally MS-observed!)
        "VWYACDEFGHIKLMN"  # 120..134
        "NEGTHRCCC"  # 135..143 (matches pos)
        "PQRSTVWYACDEFGH"  # 144..158
    )
    # Also make the 1-AA shifted overlap window of NEGAANAAA ("NNEGAANAA") match tightly
    overlap_1aa = seq[38:47]
    scorer = _DictScorer(
        {
            ("PPSPEPTID", k_i): 1.50,
            ("NEGAANAAA", k_i): 1.52,
            (overlap_1aa, k_i): 1.53,  # must be skipped by min_position_separation!
            ("NEGAWNFFF", k_i): 1.60,
            ("EMMARGDDD", k_i): 1.50,  # must be blocked by ProtectedIdentityIndex!
            ("KKSPEPEEE", k_i): 1.50,  # must be blocked by observed!
            ("NEGTHRCCC", k_i): 1.70,
        },
        default=4.5,
    )
    pos = _make_pos(
        "pos_1",
        "PPSPEPTID",
        (mhc_i,),
        "PROT1",
        seq,
        cell_type="Dendritic cell",
        cytokines={"IFNG": Exposure("present")},
    )
    protected = ProtectedIdentityIndex({"EMMARGDDD"})
    out_gz = tmp_path / "hardpairs.jsonl.gz"

    bgs, stats = mine_affinity_matched_backgrounds(
        [pos],
        {"PROT1": Protein("PROT1", "GENE1", seq)},
        scorer,
        observed={"KKSPEPEEE"},
        protected=protected,
        config=MiningConfig(
            neg_per_pos=3,
            max_candidates_per_positive=200,
            max_log10_nm_gap=0.25,
            min_position_separation=5,
            min_hamming_between_negs=3,
            seed=42,
        ),
        out_path=out_gz,
    )

    assert stats["test_accessed"] is False
    assert stats["backgrounds_emitted"] == 3
    assert stats["negatives_per_matched_positive_hist"] == {"3": 1}
    assert stats["counts"]["skip_redundant_overlap_neg"] >= 1
    assert {b.peptide for b in bgs} == {"NEGAANAAA", "NEGAWNFFF", "NEGTHRCCC"}

    # Every emitted background passes validate_pair, preserves APC context, and points to pos_1
    for bg in bgs:
        assert bg.provenance[PAIR_KEY] == "pos_1"
        assert bg.provenance["synthetic"] is True
        assert bg.context.cell_type == "Dendritic cell"
        assert bg.context.cytokines == {"IFNG": Exposure("present")}
        assert bg.context.source_n is not None and bg.context.source_c is not None
        assert validate_pair(pos, bg) == "I"

    # Verify gzipped JSONL round-trip
    with gzip.open(out_gz, "rt", encoding="utf-8") as f:
        loaded = [Observation.from_dict(json.loads(line)) for line in f]
    assert len(loaded) == 3
    for bg in loaded:
        assert validate_pair(pos, bg) == "I"


def test_multiallelic_same_winner_and_masked_flank_parity():
    mhc_a = MHCInput(groove1=G1_A, groove2=G2_A, mhc_class="I")
    mhc_b = MHCInput(groove1=G1_B, groove2=G2_B, mhc_class="I")
    ka, kb = candidate_key(mhc_a), candidate_key(mhc_b)

    seq = (
        "AAAAAAAAAAAAAAA"
        "PPSMALTII"
        "CCCCCCCCCCCCCCC"
        "WRANGWINA"  # swaps winning allele (B instead of A) -> REJECT
        "DDDDDDDDDDDDDDD"
        "WADALLELA"  # matches A, misses B by 1.5 log10 -> REJECT under profile='all'
        "EEEEEEEEEEEEEEE"
        "GAADMALTI"  # matches BOTH A and B within 0.25 -> ACCEPT
        "FFFFFFFFFFFFFFF"
    )
    scorer = _DictScorer(
        {
            ("PPSMALTII", ka): 1.50,
            ("PPSMALTII", kb): 3.50,
            ("WRANGWINA", ka): 3.50,
            ("WRANGWINA", kb): 1.50,
            ("WADALLELA", ka): 1.55,
            ("WADALLELA", kb): 2.00,
            ("GAADMALTI", ka): 1.60,
            ("GAADMALTI", kb): 3.40,
        },
        default=4.8,
    )
    # Masked ambiguous flanks on positive -> backgrounds MUST also mask all 4 flank/terminus fields
    pos = _make_pos("pos_multi", "PPSMALTII", (mhc_a, mhc_b), "PM1", seq, mask_flanks=True)
    bgs, stats = mine_affinity_matched_backgrounds(
        [pos],
        {"PM1": seq},
        scorer,
        observed=set(),
        protected=ProtectedIdentityIndex(set()),
        config=MiningConfig(
            neg_per_pos=3,
            max_candidates_per_positive=200,
            max_log10_nm_gap=0.25,
            profile="all",
        ),
    )

    assert stats["backgrounds_emitted"] == 1
    assert bgs[0].peptide == "GAADMALTI"
    assert bgs[0].context.source_n is None
    assert bgs[0].context.source_c is None
    assert bgs[0].context.n_terminus is None
    assert bgs[0].context.c_terminus is None
    assert validate_pair(pos, bgs[0]) == "I"


def test_presto_scorer_pair_pool_and_zero_binding_gradient():
    torch.manual_seed(43)
    cfg = PrestoConfig(
        endpoints=("ic50", "elution_observation"),
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_detach_affinity=False,
        elution_shared_processing=True,
        shared_processing_v2=True,
        elution_pmhc_residual=True,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        query_anchor_embedding=True,
        groove_pocket_pool=True,
        register_pooling_mode="thermodynamic",
    )
    model = Presto(cfg)
    scales = {"ic50": TargetScale(mean=2.5, scale=1.2, exact_count=100)}
    scorer = PrestoAffinityScorer(model, scales, batch_size=32, use_bf16=False)

    seq_i = (
        "ACDEFGHIKLMNPQRWWYFFCCMMVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST"
        "VWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    )
    seq_ii = (
        "GGGGGGGGGGGGGGGHHNNKKQQDDYYMMRHHHHHHHHHHHHHHHCDEFGHIKLMNPQRSTVWY"
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACD"
    )
    mhc_i = MHCInput(groove1=G1_A, groove2=G2_A, mhc_class="I")
    mhc_ii = MHCInput(groove1=G1_C2, groove2=G2_C2, mhc_class="II")

    pos_i = _make_pos("pos_cls1", "WWYFFCCMM", (mhc_i,), "P_I", seq_i, study_id="s_i")
    pos_ii = _make_pos("pos_cls2", "HHNNKKQQDDYYMMR", (mhc_ii,), "P_II", seq_ii, study_id="s_ii")

    bgs, stats = mine_affinity_matched_backgrounds(
        [pos_i, pos_ii],
        {"P_I": seq_i, "P_II": seq_ii},
        scorer,
        observed=set(),
        protected=ProtectedIdentityIndex(set()),
        config=MiningConfig(
            neg_per_pos=2,
            max_candidates_per_positive=80,
            max_log10_nm_gap=0.5,  # untrained tiny model has tight predictions around mean
            min_position_separation=5,
            min_hamming_between_negs=2,
            seed=43,
        ),
    )

    assert stats["backgrounds_by_class"]["I"] == 2
    assert stats["backgrounds_by_class"]["II"] == 2

    # Feed directly into canonical PairPool (multi-negative per positive!)
    index = [pos_i, pos_ii, *bgs]
    pool = PairPool(index)
    assert pool.counts["I"]["pairs"] == 2
    assert pool.counts["II"]["pairs"] == 2

    import random

    drawn = pool.draw(index, random.Random(43), pairs_per_class=2)
    assert len(drawn) == 4

    # Verify pair_loss backpropagates into shared_processing with EXACT 0.0 gradient into binding!
    model.train()
    model.zero_grad()
    flat_obs = [row for pair in drawn for row in pair]
    outputs = model(flat_obs)
    ploss = pair_loss(outputs, logit_weight=0.0, selection_weight=1.0, margin=0.75)
    assert torch.isfinite(ploss).item()
    ploss.backward()

    # Zero gradient into binding affinity decoder/head (`heads.ic50` / `assay_layer`)
    binding_grads = [
        p.grad.abs().max().item()
        for name, p in model.named_parameters()
        if ("heads.ic50" in name or "assay_layer" in name or "groove_pocket" in name)
        and p.grad is not None
    ]
    assert all(g == 0.0 for g in binding_grads)

    # Non-zero gradient into elution_shared (SharedProcessingModule)
    proc_grads = [
        p.grad.abs().max().item()
        for name, p in model.named_parameters()
        if "elution_shared" in name and p.grad is not None
    ]
    assert any(g > 0.0 for g in proc_grads)

    # Also verify full train_step with processing_pairs succeeds cleanly
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    res = train_step(
        model,
        opt,
        [pos_i, pos_ii],
        scales,
        processing_pairs=drawn,
        processing_pair_weight=0.9,
        processing_pair_selection_weight=1.0,
        processing_pair_margin=0.75,
    )
    assert res["loss_parts"]["processing_pairs"] == 4
    assert torch.isfinite(torch.tensor(res["loss"])).item()
