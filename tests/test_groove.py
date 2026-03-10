from presto.data.groove import (
    DEFAULT_CLASS_I_GROOVE_HALF_1_LEN,
    DEFAULT_CLASS_I_GROOVE_HALF_2_LEN,
    DEFAULT_CLASS_II_ALPHA_GROOVE_LEN,
    DEFAULT_CLASS_II_BETA_GROOVE_LEN,
    PreparedMHCInput,
    extract_groove,
    find_cys_pairs,
    parse_class_i,
    parse_class_ii_alpha,
    parse_class_ii_beta,
    prepare_mhc_input,
)


HLA_A0201 = (
    "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQL"
    "RAYLDGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
)

HLA_DRA0101 = (
    "MAISGVPVLGFFIIAVLMSAQESWAIKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLP"
    "STEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGLRKSNAAERRGPL"
)

HLA_DQA10501 = (
    "MILNKALMLGALALTTVMSPCGGEDIVADHVASYGVNLYQSYGPSGQYTHEFDGDEQFYVDLGRKETVWCLPVLRQFRFDPQFALTNIAVLKHNLNSLIKRSNSTAATNEVPEVTVFSKSPVTLGQPNILICLVDNIFPPVVNITWLSNGHSVTEGVSETSFLSKSDHSFFKISYLTLLP"
    "SAEESYDCKVEHWGLDQPLLKHWEPEIPAPMSELTETVVCALGLSVGLMGIVVGTVFIIQGLRSVGASRHQGPL"
)

HLA_DRB10101 = (
    "MVCLKLPGGSCMTALTVTLMVLSSPLALAGDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQRRVEPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFRNGQEEKAGVVSTGLIQNG"
    "DWTFQTLVMLETVPRSGEVYTCQVEHPSVTSPLTVEWRAQSESAQSKMLSGIGGFVLGLIFLGLGLFIYFRNQKGHSGLQPTGFLS"
)

HLA_DPB10401 = (
    "MMVLQVSAAPRTVALTALLMVLLTSVVQGRATPENYLFQGRQECYAFNGTQRFLERYIYNREEFARFDSDVGEFRAVTELGRPAAEYWNSQKDILEEKRAVPDRMCRHNYELGGPMTLQRRVQPRVNVSPSKKGPLQHHNLLVCHVTDFYPGSIQVRWFLNGQEETAGVVSTNLIRNGDWT"
    "FQILVMLEMTPQQGDVYTCQVEHTSLDSPVTVEWKAQSDSARSKTLTGAGGFVLGLIICGVGIFMHRRSKKVQRGSA"
)

SASA_DAA_FRAGMENT = "LHIDLVITGCSDSDGLNMYGLDGEEMWYADFNKGEGVMPLPPFADPFTYPGAYEGAVGNQGICKANLATCIKAYKNPEEKI"
SASA_DAB_FRAGMENT = "GYFYHMMRQCRYSSKDLQGIELITSYVFNQAEYIRFNSTVGKYVGYTEYGVKNAEAWNKGPELAGELGELERVCKHNAPIYYS"
GAGA_BF2_0020101 = (
    "MGPCGALGLGLLLAAVCGAAAELHTLRYIRTAMTDPGPGLPWYVDVGYVDGELFVHYNSTARRYVPRTEWIAAKADQQYWDGQTQIGQGNEQIDRENLGILQRRYNQTGGSHTVQWMYGCDILEGGPIRGYYQMAYDGRDFTAFDKGTMTFTAAVPEAVPTKRKWEEGDYAEGLKQYLEETCVEWLRRYVEYGKAELGRRERPEVRVWGKEADGILTLSCRAHGFYPRPIVVSWLKDGAVRGQDAHSGGIVPNGDGTYHTWVTIDAQPGDGDKYQCRVEHASLPQPGLYSWEPPQPNLVPIVAGVAVAIVAIAIMVGVGFIIYRRHAGKKGKGYNIAPDREGGSSSSSTGSNPAI"
)
ONMY_DAB_1601 = (
    "EPHVRLSSVTPPSGRHPAMLMCSAYDFYPKPIRVTWLRDGREVKSDVTSTEELANGDWYYQIHSHLEYTPKSGEKISCMVEHISLTEPMMYHWDPSLPEAERNKIAIGASGLVLGTILALAGLIYYKKKSS"
)


def test_find_cys_pairs_simple():
    pairs = find_cys_pairs("AAC" + ("A" * 56) + "CBBBC" + ("A" * 64) + "C")
    assert pairs == [(2, 59, 57), (2, 63, 61), (59, 128, 69), (63, 128, 65)]


def test_parse_class_i_reference_hla_a0201():
    result = parse_class_i(HLA_A0201, allele="HLA-A*02:01", gene="A")
    assert result.status == "ok"
    assert result.anchor_type == "alpha2_cys"
    assert result.anchor_cys1 == 124
    assert result.anchor_cys2 == 187
    assert result.mature_start == 23
    assert len(result.groove_half_1) == DEFAULT_CLASS_I_GROOVE_HALF_1_LEN
    assert len(result.groove_half_2) == DEFAULT_CLASS_I_GROOVE_HALF_2_LEN
    assert result.groove_half_1.startswith("AGSHSM")
    assert result.groove_half_2.startswith("GSHTVQ")


def test_parse_class_i_signal_peptide_invariance():
    full_result = parse_class_i(HLA_A0201, allele="HLA-A*02:01", gene="A")
    mature_seq = HLA_A0201[full_result.mature_start :]
    mature_result = parse_class_i(mature_seq, allele="HLA-A*02:01", gene="A")
    assert mature_result.status == "ok"
    assert mature_result.mature_start == 0
    assert mature_result.groove_half_1 == full_result.groove_half_1
    assert mature_result.groove_half_2 == full_result.groove_half_2


def test_parse_class_i_alpha3_fallback_when_alpha2_pair_removed():
    full_result = parse_class_i(HLA_A0201, allele="HLA-A*02:01", gene="A")
    assert full_result.anchor_cys1 is not None
    mutated = list(HLA_A0201)
    mutated[full_result.anchor_cys1] = "A"
    fallback = parse_class_i("".join(mutated), allele="HLA-A*02:01", gene="A")
    assert fallback.status == "alpha3_fallback"
    assert fallback.anchor_type == "alpha3_cys"
    assert len(fallback.groove_half_1) == DEFAULT_CLASS_I_GROOVE_HALF_1_LEN
    assert 80 <= len(fallback.groove_half_2) <= 100


def test_parse_class_i_chicken_bf2_reference():
    result = parse_class_i(GAGA_BF2_0020101, allele="Gaga-BF2*002:01:01", gene="BF2")
    assert result.status == "ok"
    assert result.mature_start == 18
    assert result.anchor_cys1 == 119
    assert result.anchor_cys2 == 181
    assert len(result.groove_half_1) == DEFAULT_CLASS_I_GROOVE_HALF_1_LEN
    assert len(result.groove_half_2) == 92
    assert result.groove_half_1.startswith("AAAELH")
    assert result.groove_half_2.startswith("GSHTVQ")


def test_parse_class_ii_alpha_reference_dra():
    result = parse_class_ii_alpha(HLA_DRA0101, allele="HLA-DRA*01:01", gene="DRA1")
    assert result.status == "ok"
    assert result.anchor_type == "alpha2_cys"
    assert result.anchor_cys1 == 131
    assert result.mature_start == 24
    assert len(result.groove_half_1) == DEFAULT_CLASS_II_ALPHA_GROOVE_LEN
    assert result.groove_half_1.startswith("AIKEEH")


def test_parse_class_ii_alpha_dqa_prefers_late_ig_pair():
    result = parse_class_ii_alpha(HLA_DQA10501, allele="HLA-DQA1*05:01", gene="DQA1")
    assert result.status == "ok"
    assert result.anchor_cys1 == 131
    assert result.anchor_cys2 == 187
    assert result.mature_start == 24
    assert len(result.groove_half_1) == DEFAULT_CLASS_II_ALPHA_GROOVE_LEN
    assert result.groove_half_1.startswith("DIVADH")


def test_parse_class_ii_beta_drb_ignores_signal_peptide_pairs():
    result = parse_class_ii_beta(HLA_DRB10101, allele="HLA-DRB1*01:01", gene="DRB1")
    assert result.status == "ok"
    assert result.anchor_type == "beta2_cys"
    assert result.anchor_cys1 == 145
    assert result.anchor_cys2 == 201
    assert result.secondary_cys1 == 43
    assert result.secondary_cys2 == 107
    assert result.mature_start == 28
    assert len(result.groove_half_2) == DEFAULT_CLASS_II_BETA_GROOVE_LEN
    assert result.groove_half_2.startswith("AGDTRP")


def test_parse_class_ii_beta_dpb_reference():
    result = parse_class_ii_beta(HLA_DPB10401, allele="HLA-DPB1*04:01", gene="DPB1")
    assert result.status == "ok"
    assert result.anchor_type == "beta2_cys"
    assert result.anchor_cys1 == 143
    assert result.anchor_cys2 == 199
    assert result.secondary_cys1 == 43
    assert result.secondary_cys2 == 105
    assert 92 <= len(result.groove_half_2) <= 94
    assert result.groove_half_2.startswith("VQGRAT")


def test_parse_class_ii_alpha_accepts_salmon_groove_fragment():
    result = parse_class_ii_alpha(SASA_DAA_FRAGMENT, allele="Sasa-DAA*03:03:02", gene="DAA")
    assert result.status == "fragment_fallback"
    assert result.anchor_type == "raw_fragment"
    assert result.groove_half_1 == SASA_DAA_FRAGMENT


def test_parse_class_ii_beta_accepts_salmon_groove_fragment():
    result = parse_class_ii_beta(SASA_DAB_FRAGMENT, allele="Sasa-DAB*03:02", gene="DAB")
    assert result.status == "fragment_fallback"
    assert result.anchor_type == "raw_fragment"
    assert result.groove_half_2 == SASA_DAB_FRAGMENT


def test_parse_class_ii_beta_trout_beta1_only_fallback():
    result = parse_class_ii_beta(ONMY_DAB_1601, allele="Onmy-DAB*16:01", gene="DAB")
    assert result.status == "beta1_only_fallback"
    assert result.anchor_type == "beta1_cys"
    assert result.anchor_cys1 == 21
    assert result.anchor_cys2 == 77
    assert result.mature_start == 6
    assert len(result.groove_half_2) == 86
    assert result.groove_half_2.startswith("SSVTPP")


def test_extract_groove_dispatch():
    class_i = extract_groove(HLA_A0201, mhc_class="I", allele="HLA-A*02:01", gene="A")
    class_ii_alpha = extract_groove(
        HLA_DRA0101,
        mhc_class="II",
        chain="alpha",
        allele="HLA-DRA*01:01",
        gene="DRA",
    )
    class_ii_beta = extract_groove(
        HLA_DRB10101,
        mhc_class="II",
        chain="beta",
        allele="HLA-DRB1*01:01",
        gene="DRB1",
    )
    assert class_i.status == "ok"
    assert class_ii_alpha.status == "ok"
    assert class_ii_beta.status == "ok"


def test_extract_groove_infers_class_ii_chain_from_name_first():
    alpha = extract_groove(
        HLA_DRA0101,
        mhc_class="II",
        allele="HLA-DRA*01:01",
    )
    beta = extract_groove(
        HLA_DRB10101,
        mhc_class="II",
        allele="HLA-DRB1*01:01",
    )
    assert alpha.status == "ok"
    assert alpha.chain == "alpha"
    assert beta.status == "ok"
    assert beta.chain == "beta"


def test_extract_groove_class_ii_without_name_or_chain_fails_explicitly():
    result = extract_groove(
        HLA_DRA0101,
        mhc_class="II",
    )
    assert result.status == "ambiguous_chain"
    assert "alpha_status=ok" in result.flags
    assert "beta_status=ok" in result.flags


def test_extract_groove_class_ii_can_use_sequence_only_when_unique():
    result = extract_groove(
        ONMY_DAB_1601,
        mhc_class="II",
    )
    assert result.status == "beta1_only_fallback"
    assert result.chain == "beta"


def test_prepare_mhc_input_class_i_and_class_ii():
    class_i = prepare_mhc_input(mhc_a=HLA_A0201, mhc_class="I")
    class_ii = prepare_mhc_input(
        mhc_a=HLA_DRA0101,
        mhc_b=HLA_DRB10101,
        mhc_class="II",
    )
    assert isinstance(class_i, PreparedMHCInput)
    assert class_i.used_fallback is False
    assert len(class_i.groove_half_1) == DEFAULT_CLASS_I_GROOVE_HALF_1_LEN
    assert len(class_i.groove_half_2) == DEFAULT_CLASS_I_GROOVE_HALF_2_LEN
    assert class_ii.used_fallback is False
    assert len(class_ii.groove_half_1) == DEFAULT_CLASS_II_ALPHA_GROOVE_LEN
    assert len(class_ii.groove_half_2) == DEFAULT_CLASS_II_BETA_GROOVE_LEN


def test_prepare_mhc_input_fallback_when_parsing_fails():
    prepared = prepare_mhc_input(
        mhc_a="A" * 200,
        mhc_b="B" * 200,
        mhc_class="II",
        allow_fallback_truncation=True,
    )
    assert prepared.used_fallback is True
    assert len(prepared.groove_half_1) == DEFAULT_CLASS_II_ALPHA_GROOVE_LEN
    assert len(prepared.groove_half_2) == DEFAULT_CLASS_II_BETA_GROOVE_LEN
