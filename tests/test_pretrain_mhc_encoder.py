from presto.scripts.pretrain_mhc_encoder import _is_valid_mhc_segment


def test_is_valid_mhc_segment_rejects_non_amino_acid_tokens():
    assert _is_valid_mhc_segment("ACDEFGHIKLMNPQRSTVWYX")
    assert not _is_valid_mhc_segment("ACDE?GHIK")
    assert not _is_valid_mhc_segment("")
