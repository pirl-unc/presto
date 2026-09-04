"""Tests for the hitlist-backed record adapter.

These run without the real ``hitlist`` package by injecting a stub module that
returns a caller-supplied frame from ``generate_training_table``.
"""

import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from presto.data.hitlist_source import (  # noqa: E402
    BINDING_COLUMNS,
    MS_COLUMNS,
    PROTEIN_MAPPING_COLUMNS,
    SHARED_COLUMNS,
    assert_columns_present,
    training_columns,
    _clean,
    _collapse_source_mappings,
    _method_counts,
    _qualifier_from_inequality,
    _select_best_mapping,
    _split_allele_set,
    load_records_from_hitlist,
    normalize_ingested_peptide,
)
from presto.data.vocab import drop_unencodable_sequence  # noqa: E402

IC50 = "half maximal inhibitory concentration (IC50)"


def _binding_row(**overrides):
    row = {
        "peptide": "SIINFEKLA",
        "mhc_restriction": "HLA-A*02:01",
        "mhc_allele_set": "HLA-A*02:01",
        "mhc_class": "I",
        "host": "Homo sapiens (human)",
        "source_organism": "Homo sapiens",
        "source": "iedb",
        "assay_iri": "http://www.iedb.org/assay/11074",
        "reference_iri": "http://www.iedb.org/reference/42",
        "pmid": "12345678",
        "n_flank": "AAAAAAAAAA",
        "c_flank": "CCCCCCCCCC",
        "position": 10,
        "evidence_row_id": "row-1",
        "gene_name": "GENE1",
        "gene_id": "ENSG1",
        "protein_id": "ENSP1",
        "transcript_id": "ENST1",
        "is_canonical_transcript": True,
        "proteome": "Homo sapiens",
        "proteome_source": "species",
        "response_measured": IC50,
        "quantitative_value": 25.0,
        "measurement_units": "nM",
        "measurement_inequality": "=",
        "assay_method": "purified MHC/competitive/radioactivity",
    }
    row.update(overrides)
    return row


def _ms_row(**overrides):
    row = {
        "peptide": "SIINFEKLA",
        "mhc_restriction": "HLA-A*02:01",
        "mhc_allele_set": "HLA-A*02:01;HLA-B*07:02",
        "mhc_class": "I",
        "host": "Homo sapiens (human)",
        "source_organism": "Homo sapiens",
        "source": "iedb",
        "assay_iri": "http://www.iedb.org/assay/22001",
        "reference_iri": "http://www.iedb.org/reference/84",
        "pmid": "87654321",
        "n_flank": "GGGGGGGGGG",
        "c_flank": "TTTTTTTTTT",
        "position": 10,
        "evidence_row_id": "ms-1",
        "gene_name": "GENE1",
        "gene_id": "ENSG1",
        "protein_id": "ENSP1",
        "transcript_id": "ENST1",
        "is_canonical_transcript": True,
        "proteome": "Homo sapiens",
        "proteome_source": "species",
        "cell_line_name": "HeLa",
        "source_tissue": "Skin",
    }
    row.update(overrides)
    return row


def _install_stub_hitlist(monkeypatch, binding_rows, ms_rows):
    module = types.ModuleType("hitlist")

    def generate_training_table(*, include_evidence, columns=None, **_kwargs):
        rows = binding_rows if include_evidence == "binding" else ms_rows
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=columns or [])
        # Real hitlist returns every column it has that was asked for; a
        # fixture row only spells out the fields that test cares about. Fill
        # the rest in as empty so the stub models "hitlist has this column and
        # it happens to be blank" rather than "hitlist dropped it", which is
        # what `assert_columns_present` is there to catch.
        for column in columns or []:
            if column not in frame.columns:
                frame[column] = ""
        return frame

    module.generate_training_table = generate_training_table
    monkeypatch.setitem(sys.modules, "hitlist", module)


class TestHelpers:
    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), pd.NA, "", "   "],
        ids=["none", "float-nan", "pandas-na", "empty", "whitespace"],
    )
    def test_nullable_sequences_never_become_fake_amino_acids(self, value):
        assert drop_unencodable_sequence(value) == ""
        assert normalize_ingested_peptide(value) == ""

    def test_real_lowercase_nan_peptide_remains_distinct_from_float_nan(self):
        assert normalize_ingested_peptide("nan") == "NAN"

    def test_qualifier_from_inequality(self):
        assert _qualifier_from_inequality("<") == -1
        assert _qualifier_from_inequality("<=") == -1
        assert _qualifier_from_inequality(">") == 1
        assert _qualifier_from_inequality(">=") == 1
        assert _qualifier_from_inequality("=") == 0
        assert _qualifier_from_inequality("") == 0
        assert _qualifier_from_inequality(None) == 0

    def test_split_allele_set(self):
        assert _split_allele_set("HLA-A*02:01") == ["HLA-A*02:01"]
        assert _split_allele_set("HLA-A*02:01;HLA-B*07:02") == [
            "HLA-A*02:01",
            "HLA-B*07:02",
        ]
        assert _split_allele_set(["HLA-A*02:01"]) == ["HLA-A*02:01"]
        assert _split_allele_set("") == []
        assert _split_allele_set(None) == []
        assert _split_allele_set(float("nan")) == []
        assert _split_allele_set(pd.NA) == []
        assert _split_allele_set(["HLA-A*02:01", pd.NA, float("nan")]) == ["HLA-A*02:01"]

    @pytest.mark.parametrize("value", [None, float("nan"), pd.NA, ""])
    def test_null_qualifiers_are_exact(self, value):
        assert _qualifier_from_inequality(value) == 0

    def test_clean_normalizes_missing_markers(self):
        assert _clean("nan") == ""
        assert _clean(None) == ""
        assert _clean("  X  ") == "X"

    def test_select_best_mapping_prefers_canonical_transcript(self):
        frame = pd.DataFrame(
            [
                {
                    "evidence_row_id": "a",
                    "gene_name": "G",
                    "protein_id": "P2",
                    "transcript_id": "T2",
                    "is_canonical_transcript": False,
                    "n_flank": "NO",
                },
                {
                    "evidence_row_id": "a",
                    "gene_name": "G",
                    "protein_id": "P1",
                    "transcript_id": "T1",
                    "is_canonical_transcript": True,
                    "n_flank": "YES",
                },
            ]
        )
        collapsed = _select_best_mapping(frame, source_mapping_policy="legacy_global_canonical")
        assert len(collapsed) == 1
        assert collapsed.iloc[0]["n_flank"] == "YES"
        assert collapsed.iloc[0]["source_mapping_category"] == "within_gene_canonical"

    def test_legacy_policy_keeps_cross_gene_choice_for_control_only(self):
        frame = pd.DataFrame(
            [
                _binding_row(
                    gene_name="GENE2",
                    protein_id="P2",
                    transcript_id="T2",
                    n_flank="BBBB",
                    c_flank="DDDD",
                ),
                _binding_row(
                    gene_name="GENE1",
                    protein_id="P1",
                    transcript_id="T1",
                    n_flank="AAAA",
                    c_flank="CCCC",
                ),
            ]
        )
        masked, masked_stats = _collapse_source_mappings(frame)
        legacy, legacy_stats = _collapse_source_mappings(
            frame, source_mapping_policy="legacy_global_canonical"
        )

        # Masking clears the junction rather than writing a sentinel residue:
        # the model supplies the unknown-context embedding by padding the flank
        # window with `?`, so no marker belongs in the sequence string.
        assert masked.iloc[0]["n_flank"] == masked.iloc[0]["c_flank"] == ""
        assert legacy.iloc[0]["n_flank"] == "AAAA"
        assert legacy.iloc[0]["c_flank"] == "CCCC"
        assert masked.iloc[0]["source_mapping_category"] == "cross_gene_unresolved"
        assert legacy.iloc[0]["source_mapping_category"] == "cross_gene_unresolved"
        assert masked_stats["source_mapping_policy"] == "mask_unresolved"
        assert legacy_stats["source_mapping_policy"] == "legacy_global_canonical"

    def test_unknown_mapping_policy_is_rejected(self):
        with pytest.raises(ValueError, match="source_mapping_policy"):
            _collapse_source_mappings(
                pd.DataFrame([_binding_row()]), source_mapping_policy="coin_flip"
            )

    def test_method_counts_labels_missing_as_unspecified(self):
        class Rec:
            def __init__(self, method):
                self.assay_method = method

        counts = _method_counts([Rec("a"), Rec("a"), Rec(None)])
        assert counts == {"a": 2, "unspecified": 1}


class TestRouting:
    def test_ingest_stats_preserve_pre_cap_counts_and_drop_reasons(self, monkeypatch):
        rows = [
            _binding_row(evidence_row_id="row-1"),
            _binding_row(evidence_row_id="row-2", peptide="GILGFVFTL"),
        ]
        _install_stub_hitlist(monkeypatch, rows, [])

        binding, _, _, _, _, _, _, stats = load_records_from_hitlist(
            max_binding=1,
            sampling_seed=59,
        )

        assert len(binding) == 1
        assert stats["sampling_seed"] == 59
        assert stats["counts_before_cap"]["binding"] == 2
        assert stats["counts"]["binding"] == 1
        assert stats["rows_dropped_by_cap"]["binding"] == 1
        assert stats["requested_caps"]["binding"] == 1

    def test_affinity_row_becomes_binding_record_with_flanks(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row()], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert len(binding) == 1
        record = binding[0]
        assert record.value == 25.0
        assert record.measurement_type == IC50
        assert record.flank_n == "AAAAAAAAAA"
        assert record.flank_c == "CCCCCCCCCC"
        assert record.source_mapping_category == "single"
        assert record.source_mapping_n_candidates == 1
        assert record.flank_context_resolved is True
        assert record.evidence_row_id == "row-1"
        assert record.assay_iri.endswith("/11074")
        assert record.reference_iri.endswith("/42")
        assert record.pmid == "12345678"
        assert record.mapping_gene_id == "ENSG1"
        assert record.mapping_protein_id == "ENSP1"
        assert record.mapping_transcript_id == "ENST1"
        assert record.mapping_position == 10
        assert record.mapping_proteome == "Homo sapiens"
        assert stats["flank_coverage"]["binding"] == 1.0

    def test_cross_gene_disagreement_keeps_label_but_masks_flanks(self, monkeypatch):
        rows = [
            _binding_row(
                gene_name="GENE1",
                gene_id="ENSG1",
                protein_id="ENSP1",
                transcript_id="ENST1",
                n_flank="AAAAAAAAAA",
                c_flank="CCCCCCCCCC",
            ),
            _binding_row(
                gene_name="GENE2",
                gene_id="ENSG2",
                protein_id="ENSP2",
                transcript_id="ENST2",
                n_flank="GGGGGGGGGG",
                c_flank="TTTTTTTTTT",
            ),
        ]
        _install_stub_hitlist(monkeypatch, rows, [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert len(binding) == 1
        record = binding[0]
        assert record.value == 25.0
        assert record.flank_n == record.flank_c == ""
        # Not a terminus: an unresolved junction must not be read as "the
        # protein ended here". The masking marker suppresses terminus inference
        # without deleting the selected mapping from the lineage record.
        assert record.flank_n_is_terminus is False
        assert record.flank_c_is_terminus is False
        assert record.mapping_position == 10
        assert record.source_mapping_category == "cross_gene_unresolved"
        assert record.flank_context_resolved is False
        assert stats["mapping_ambiguity"]["binding"]["category_counts"] == {
            "cross_gene_unresolved": 1
        }

        from presto.data.collate import PrestoCollator
        from presto.data.loaders import PrestoDataset

        dataset = PrestoDataset(binding_records=binding, strict_mhc_resolution=False)
        sample = dataset[0]
        assert sample.bind_value == 25.0
        assert sample.source_mapping_category == "cross_gene_unresolved"
        assert sample.sample_id == "bind:row-1"
        assert sample.evidence_row_id == "row-1"
        assert sample.source_mhc_alleles == ("HLA-A*02:01",)
        batch = PrestoCollator()([sample])
        assert batch.bind_mask.tolist() == [1.0]
        assert batch.source_mapping_categories == ["cross_gene_unresolved"]
        # Host-side lists, not tensors: these are diagnostics that never enter
        # `model.forward`, so moving them to the device and back was pure cost.
        assert batch.source_mapping_n_candidates == [2]
        assert batch.flank_context_resolved == [False]
        assert batch.source_lineage["peptide"] == ["SIINFEKLA"]
        assert batch.source_lineage["source_mhc_alleles"] == ["HLA-A*02:01"]
        assert batch.source_lineage["resolved_mhc_alleles"] == ["HLA-A*02:01"]
        assert batch.source_lineage["mapping_protein_id"] == ["ENSP1"]
        assert batch.source_lineage["source_sample_label"] == [""]
        moved = batch.to("cpu")
        assert moved.source_mapping_categories == ["cross_gene_unresolved"]
        assert moved.source_mapping_n_genes == [2]
        assert moved.source_lineage == batch.source_lineage

    def test_policies_filter_selected_x_before_masking(self, monkeypatch):
        """Policy changes flank input, never which supervised rows survive."""
        rows = [
            _binding_row(
                gene_name="GENE1",
                gene_id="ENSG1",
                protein_id="ENSP1",
                transcript_id="ENST1",
                n_flank="XAAAAAAAAA",
                c_flank="CCCCCCCCCC",
            ),
            _binding_row(
                gene_name="GENE2",
                gene_id="ENSG2",
                protein_id="ENSP2",
                transcript_id="ENST2",
                n_flank="GGGGGGGGGG",
                c_flank="TTTTTTTTTT",
                is_canonical_transcript=False,
            ),
        ]
        _install_stub_hitlist(monkeypatch, rows, [])
        masked, *_ = load_records_from_hitlist(source_mapping_policy="mask_unresolved")
        legacy, *_ = load_records_from_hitlist(source_mapping_policy="legacy_global_canonical")

        assert masked == legacy == []

    def test_nan_value_is_skipped(self, monkeypatch):
        """float(nan) does not raise, so NaN must be tested for explicitly.

        Without this guard a missing measurement enters training as a NaN
        target and silently poisons the loss.
        """
        _install_stub_hitlist(monkeypatch, [_binding_row(quantitative_value=float("nan"))], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_no_numeric_value"] == 1

    def test_unexpected_unit_is_skipped(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row(measurement_units="uM")], [])
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_unexpected_unit"] == 1

    def test_unroutable_response_is_counted_not_ingested(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [_binding_row(response_measured="qualitative binding", measurement_units="")],
            [],
        )
        binding, _, _, _, _, _, _, stats = load_records_from_hitlist()
        assert binding == []
        assert stats["skipped_unroutable_response"] == 1

    def test_half_life_converts_minutes_to_hours(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [
                _binding_row(
                    response_measured="half life",
                    measurement_units="min",
                    quantitative_value=120.0,
                    assay_method="purified MHC/direct/radioactivity",
                )
            ],
            [],
        )
        _, _, stability, _, _, _, _, stats = load_records_from_hitlist()
        assert len(stability) == 1
        assert stability[0].t_half == pytest.approx(2.0)
        assert stability[0].assay_method == "purified MHC/direct/radioactivity"
        assert stats["stability_assay_methods"] == {"purified MHC/direct/radioactivity": 1}

    def test_tm_and_kinetics_routing(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [
                _binding_row(
                    response_measured="50% dissociation temperature",
                    measurement_units="°C",
                    quantitative_value=52.0,
                    evidence_row_id="tm-1",
                ),
                _binding_row(
                    response_measured="off rate",
                    measurement_units="1/s",
                    quantitative_value=0.01,
                    evidence_row_id="koff-1",
                ),
            ],
            [],
        )
        _, kinetics, stability, _, _, _, _, _ = load_records_from_hitlist()
        assert len(stability) == 1 and stability[0].tm == 52.0
        assert len(kinetics) == 1 and kinetics[0].koff == 0.01

    def test_ms_row_becomes_elution_record_with_allele_bag(self, monkeypatch):
        _install_stub_hitlist(
            monkeypatch,
            [],
            [_ms_row(sample_label="HeLa-A02", sample_attribution="monoallelic")],
        )
        _, _, _, _, elution, _, _, stats = load_records_from_hitlist()
        assert len(elution) == 1
        assert elution[0].alleles == ["HLA-A*02:01", "HLA-B*07:02"]
        assert elution[0].source_alleles == ("HLA-A*02:01", "HLA-B*07:02")
        assert elution[0].flank_n == "GGGGGGGGGG"
        assert stats["flank_coverage"]["elution"] == 1.0

        from presto.data.collate import PrestoCollator
        from presto.data.loaders import PrestoDataset

        sample = PrestoDataset(elution_records=elution, strict_mhc_resolution=False)[0]
        batch = PrestoCollator()([sample])
        assert batch.source_lineage["source_sample_label"] == ["HeLa-A02"]
        assert batch.source_lineage["source_sample_attribution"] == ["monoallelic"]
        assert batch.source_lineage["source_mhc_alleles"] == ["HLA-A*02:01;HLA-B*07:02"]
        assert batch.source_lineage["resolved_mhc_alleles"] == ["HLA-A*02:01;HLA-B*07:02"]

    def test_ms_row_without_alleles_is_dropped(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [], [_ms_row(mhc_allele_set="", mhc_restriction="")])
        _, _, _, _, elution, _, _, _ = load_records_from_hitlist()
        assert elution == []

    def test_modalities_hitlist_does_not_carry_are_empty(self, monkeypatch):
        _install_stub_hitlist(monkeypatch, [_binding_row()], [_ms_row()])
        _, _, _, processing, _, tcell, tcr, stats = load_records_from_hitlist()
        assert processing == [] and tcell == [] and tcr == []
        assert stats["counts"]["processing"] == 0


class TestColumnContract:
    """The set of columns Presto asks hitlist for.

    hitlist projects by intersection (`export._project_training_columns` keeps
    `[c for c in requested if c in df.columns]`), so a column renamed or
    withdrawn upstream comes back absent rather than raising. Nothing else in
    the pipeline notices: the feature built from it becomes a constant and the
    loss barely moves. These tests are the place that notices.
    """

    def test_evidence_families_extend_the_shared_set(self):
        assert set(SHARED_COLUMNS) <= set(BINDING_COLUMNS)
        assert set(SHARED_COLUMNS) <= set(MS_COLUMNS)

    def test_constants_are_immutable(self):
        # A public list would let a caller append to the contract in place and
        # change what every later load requests.
        for columns in (SHARED_COLUMNS, BINDING_COLUMNS, MS_COLUMNS):
            assert isinstance(columns, tuple)
        assert isinstance(PROTEIN_MAPPING_COLUMNS, frozenset)

    def test_no_duplicate_columns(self):
        for columns in (BINDING_COLUMNS, MS_COLUMNS):
            assert len(columns) == len(set(columns))

    def test_training_columns_returns_a_fresh_mutable_list(self):
        first = training_columns("ms", include_flanks=True)
        first.append("scratch")
        assert "scratch" not in training_columns("ms", include_flanks=True)

    def test_flanks_are_dropped_only_when_not_mapping_proteins(self):
        with_flanks = training_columns("ms", include_flanks=True)
        without = training_columns("ms", include_flanks=False)
        assert PROTEIN_MAPPING_COLUMNS <= set(with_flanks)
        assert not PROTEIN_MAPPING_COLUMNS & set(without)
        assert set(with_flanks) - set(without) == PROTEIN_MAPPING_COLUMNS

    def test_unknown_evidence_family_raises(self):
        # Silently projecting the wrong columns for a whole run is worse than
        # a KeyError at the call site.
        with pytest.raises(KeyError):
            training_columns("elution", include_flanks=True)

    def test_apm_state_columns_are_per_sample_not_study_level(self):
        """The study roll-up ORs across a deposit, so a WT control inside a
        knockout study inherits the flag (pirl-unc/hitlist#353)."""
        assert "apm_genes_perturbed" in MS_COLUMNS
        assert "condition_category" in MS_COLUMNS
        assert "study_apm_perturbed" not in MS_COLUMNS
        assert "study_apm_genes" not in MS_COLUMNS

    def test_assert_columns_present_accepts_a_complete_frame(self):
        frame = pd.DataFrame(columns=training_columns("ms", include_flanks=True))
        assert_columns_present(frame, "ms", include_flanks=True)

    def test_assert_columns_present_names_a_renamed_column(self):
        columns = training_columns("ms", include_flanks=True)
        columns.remove("apm_genes_perturbed")
        frame = pd.DataFrame(columns=columns)
        with pytest.raises(RuntimeError, match="apm_genes_perturbed"):
            assert_columns_present(frame, "ms", include_flanks=True)

    def test_assert_columns_present_tolerates_extra_columns(self):
        # hitlist always adds `evidence_kind`, and grows new columns between
        # releases; only absence is a problem.
        frame = pd.DataFrame(
            columns=training_columns("binding", include_flanks=False) + ["evidence_kind"]
        )
        assert_columns_present(frame, "binding", include_flanks=False)

    def test_flank_columns_absent_is_fine_when_not_requested(self):
        frame = pd.DataFrame(columns=training_columns("ms", include_flanks=False))
        assert_columns_present(frame, "ms", include_flanks=False)

    def test_guard_reports_without_importing_hitlist(self, monkeypatch):
        """The guard must not need hitlist importable to report a problem.

        hitlist is an optional extra and this whole module tests against a
        stub, so an `import hitlist` on the failure path replaces the real
        diagnostic with a ModuleNotFoundError -- which is what happened in CI
        the first time around.

        Binding the name to None rather than deleting it is what makes this
        reproduce on a machine that *does* have hitlist installed: a None entry
        in `sys.modules` makes `import hitlist` raise, where a missing entry
        would just re-import it and the test would pass locally while failing
        in CI. Which is exactly the trap being closed.
        """
        monkeypatch.setitem(sys.modules, "hitlist", None)
        columns = training_columns("ms", include_flanks=True)
        columns.remove("condition_category")
        frame = pd.DataFrame(columns=columns)
        with pytest.raises(RuntimeError, match="condition_category"):
            assert_columns_present(frame, "ms", include_flanks=True)

    def test_guard_names_the_installed_hitlist_version(self, monkeypatch):
        stub = types.ModuleType("hitlist")
        stub.__version__ = "9.9.9"
        monkeypatch.setitem(sys.modules, "hitlist", stub)
        frame = pd.DataFrame(columns=["peptide"])
        with pytest.raises(RuntimeError, match="9.9.9"):
            assert_columns_present(frame, "ms", include_flanks=False)


class TestTheCollapseKeepsTheRowThatActuallyMapped:
    """A left-join miss must not outrank a real mapping.

    `map_source_proteins=True` emits one row per (evidence row, mapping), and
    an evidence row with no mapping still comes back -- with every identifier
    empty. Empty strings sort first ascending, so the blank candidate won
    `drop_duplicates(keep="first")` and the real flanks were discarded, while
    the summary -- computed over `_mapping_present_mask`, which had excluded
    that blank row -- still stamped the result `single` and resolved.

    A canonical flag on the mapped row masked this, which is why the cases
    below vary it: the plain non-canonical case is the one that failed.
    """

    @staticmethod
    def _collapse(canonical_real, canonical_blank):
        from presto.data.hitlist_source import _collapse_source_mappings

        frame = pd.DataFrame(
            [
                {
                    "evidence_row_id": "e1",
                    "gene_name": "G1",
                    "gene_id": "G1",
                    "protein_id": "P1",
                    "transcript_id": "T1",
                    "position": 10,
                    "n_flank": "AAAA",
                    "c_flank": "CCCC",
                    "is_canonical_transcript": canonical_real,
                },
                {
                    "evidence_row_id": "e1",
                    "gene_name": "",
                    "gene_id": "",
                    "protein_id": "",
                    "transcript_id": "",
                    "position": None,
                    "n_flank": "",
                    "c_flank": "",
                    "is_canonical_transcript": canonical_blank,
                },
            ]
        )
        collapsed, _ = _collapse_source_mappings(
            frame, source_mapping_policy="legacy_global_canonical"
        )
        return collapsed.iloc[0]

    @pytest.mark.parametrize(
        "canonical_real,canonical_blank",
        [
            (True, False),
            (False, False),
            (False, True),
            # Float-encoded flag: what a bool column with nulls becomes on a
            # parquet or CSV round trip. This case failed twice over, because
            # `_canonical_mask` also read 1.0 as False.
            (1.0, 0.0),
        ],
    )
    def test_the_mapped_candidate_wins(self, canonical_real, canonical_blank):
        row = self._collapse(canonical_real, canonical_blank)
        assert row["n_flank"] == "AAAA"
        assert row["c_flank"] == "CCCC"
        assert row["protein_id"] == "P1"

    def test_an_evidence_row_with_only_misses_still_survives(self):
        """Sorting, not filtering: the row must not vanish."""
        from presto.data.hitlist_source import _collapse_source_mappings

        frame = pd.DataFrame(
            [
                {
                    "evidence_row_id": "e2",
                    "gene_name": "",
                    "gene_id": "",
                    "protein_id": "",
                    "transcript_id": "",
                    "position": None,
                    "n_flank": "",
                    "c_flank": "",
                    "is_canonical_transcript": False,
                }
            ]
        )
        collapsed, _ = _collapse_source_mappings(
            frame, source_mapping_policy="legacy_global_canonical"
        )
        assert len(collapsed) == 1
        assert collapsed.iloc[0]["source_mapping_category"] == "unmapped"
        assert not bool(collapsed.iloc[0]["flank_context_resolved"])


class TestTheCanonicalFlagIsReadInEveryEncoding:
    """`astype(str)` turns 1.0 into "1.0", which is in no truthy set.

    A bool column carrying nulls arrives from parquet or CSV as float64, so
    every canonical row would read as non-canonical, `usable_canonical` would
    never be true, and every within-gene disagreement would degrade to
    `within_gene_unresolved`. It also disagreed with the scalar
    `flank_selection._truthy`, which returns True for 1.0.
    """

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([1.0, 0.0], [True, False]),
            ([1, 0], [True, False]),
            (["true", "false"], [True, False]),
            (["yes", "no"], [True, False]),
            ([True, False], [True, False]),
            ([1.0, "true", 0], [True, True, False]),
        ],
    )
    def test_truthiness_matches_the_scalar_helper(self, values, expected):
        from presto.data.hitlist_source import _canonical_mask

        mask = _canonical_mask(pd.DataFrame({"is_canonical_transcript": values}))
        assert mask.tolist() == expected

    def test_a_missing_flag_is_not_canonical(self):
        from presto.data.hitlist_source import _canonical_mask

        mask = _canonical_mask(pd.DataFrame({"is_canonical_transcript": [None, float("nan")]}))
        assert mask.tolist() == [False, False]


class TestMaskingIsAPureTransform:
    """It is handed a slice, and it used to write through it.

    `drop_unresolved_flank_rows` returns `frame.loc[~mask]` when rows were
    dropped and the caller's own object when none were, so assigning into the
    argument was either chained assignment or a silent rewrite of the caller's
    data.
    """

    @staticmethod
    def _frame():
        return pd.DataFrame(
            [
                {
                    "evidence_row_id": "e1",
                    "source_mapping_category": "cross_gene_unresolved",
                    "n_flank": "AAAA",
                    "c_flank": "CCCC",
                    "position": 10.0,
                }
            ]
        )

    def test_the_caller_s_frame_is_left_alone(self):
        from presto.data.hitlist_source import _mask_unresolved_mapping_context

        original = self._frame()
        masked = _mask_unresolved_mapping_context(original)
        assert original.iloc[0]["n_flank"] == "AAAA", "input was mutated"
        assert masked.iloc[0]["n_flank"] == ""

    def test_mapping_position_is_preserved_but_context_is_marked_masked(self):
        """Mask model input without destroying the chosen mapping lineage."""
        from presto.data.hitlist_source import _flank_fields, _mask_unresolved_mapping_context

        masked = _mask_unresolved_mapping_context(self._frame())
        row = masked.iloc[0]
        assert row["position"] == 10.0
        assert bool(row["source_mapping_context_masked"])
        assert _flank_fields(row) == {
            "flank_n": "",
            "flank_c": "",
            "flank_n_is_terminus": False,
            "flank_c_is_terminus": False,
        }

    def test_a_frame_without_categories_is_returned_unchanged(self):
        """`_collapse_source_mappings` no-ops when there is no `evidence_row_id`.

        The masking step then received a frame with no `source_mapping_category`
        and raised `KeyError` instead of no-opping to match.
        """
        from presto.data.hitlist_source import _mask_unresolved_mapping_context

        frame = pd.DataFrame([{"n_flank": "AAAA", "c_flank": "CCCC"}])
        assert _mask_unresolved_mapping_context(frame).equals(frame)


class TestResolvedIsNaNSafe:
    """`bool(float("nan"))` is True, so an unmatched join stamped rows resolved."""

    def test_a_nan_flag_reads_as_unresolved(self):
        from presto.data.hitlist_source import _mapping_fields

        fields = _mapping_fields({"flank_context_resolved": float("nan")})
        assert fields["flank_context_resolved"] is False

    def test_a_real_flag_still_reads_through(self):
        from presto.data.hitlist_source import _mapping_fields

        assert _mapping_fields({"flank_context_resolved": True})["flank_context_resolved"] is True
        assert _mapping_fields({})["flank_context_resolved"] is False


class TestMaskingUnmappedRowsChangesNothing:
    """`unmapped` joined the unresolved set; that must not move any data.

    Deriving resolved/unresolved from one table put `unmapped` in the
    unresolved set for the first time, so the masking step now visits those
    rows. It is a no-op by construction and this pins the construction:
    `unmapped` means no candidate was *present*, and `_mapping_present_mask`
    counts a non-empty flank as present -- so an unmapped row has no flank to
    clear. If that ever stops holding, the two policies stop being paired.
    """

    @staticmethod
    def _frame():
        return pd.DataFrame(
            [
                {
                    "evidence_row_id": "mapped",
                    "gene_name": "G1",
                    "gene_id": "G1",
                    "protein_id": "P1",
                    "transcript_id": "T1",
                    "position": 10,
                    "n_flank": "AAAA",
                    "c_flank": "CCCC",
                    "is_canonical_transcript": True,
                },
                {
                    "evidence_row_id": "missed",
                    "gene_name": "",
                    "gene_id": "",
                    "protein_id": "",
                    "transcript_id": "",
                    "position": None,
                    "n_flank": "",
                    "c_flank": "",
                    "is_canonical_transcript": False,
                },
            ]
        )

    def test_the_two_policies_agree_on_unmapped_rows(self):
        from presto.data.hitlist_source import _collapse_source_mappings

        legacy, _ = _collapse_source_mappings(
            self._frame(), source_mapping_policy="legacy_global_canonical"
        )
        masked, _ = _collapse_source_mappings(
            self._frame(), source_mapping_policy="mask_unresolved"
        )
        legacy = legacy.set_index("evidence_row_id")
        masked = masked.set_index("evidence_row_id")
        assert legacy.loc["missed", "source_mapping_category"] == "unmapped"
        for column in ("n_flank", "c_flank"):
            assert legacy.loc["missed", column] == masked.loc["missed", column] == ""
        # The resolved row is untouched by either policy.
        for column in ("n_flank", "c_flank"):
            assert legacy.loc["mapped", column] == masked.loc["mapped", column]

    def test_an_unmapped_row_cannot_carry_a_flank(self):
        """The invariant the no-op rests on."""
        from presto.data.hitlist_source import _mapping_present_mask

        frame = pd.DataFrame([{"n_flank": "AAAA", "c_flank": "", "position": None}])
        assert _mapping_present_mask(frame).tolist() == [True], (
            "a row with a flank must count as present, so it can never be classified unmapped"
        )
