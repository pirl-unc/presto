"""Tests for dataset loaders."""

import csv
from collections import Counter

import pytest

from presto.data.collate import PrestoCollator
from presto.data.groove import prepare_mhc_input
from presto.data.loaders import load_iedb_stability, load_iedb_tcell
from presto.data.loaders import load_iedb_binding, load_iedb_elution, load_iedb_processing
from presto.data.loaders import load_iedb_kinetics
from presto.data.loaders import (
    load_iedb_bcell,
    load_10x_vdj,
    load_vdjdb,
    load_vdjdb_tcr_evidence,
    load_mcpas_tcr_evidence,
)
from presto.data.loaders import (
    BindingRecord,
    ElutionRecord,
    KineticsRecord,
    PrestoDataset,
    ProcessingRecord,
    StabilityRecord,
    TCellRecord,
    Sc10xVDJRecord,
    VDJdbRecord,
    create_dataloader,
)
MHC_ALPHA_SEQ = (
    "MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQL"
    "RAYLDGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWE"
)
MHC_ALT_SEQ = MHC_ALPHA_SEQ
MHC_DR_ALPHA_SEQ = (
    "MAISGVPVLGFFIIAVLMSAQESWAIKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLP"
    "STEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGLRKSNAAERRGPL"
)
MHC_DR_BETA_SEQ = (
    "MVCLKLPGGSCMTALTVTLMVLSSPLALAGDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQRRVEPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFRNGQEEKAGVVSTGLIQNG"
    "DWTFQTLVMLETVPRSGEVYTCQVEHPSVTSPLTVEWRAQSESAQSKMLSGIGGFVLGLIFLGLGLFIYFRNQKGHSGLQPTGFLS"
)
MHC_SHORT_SEQ = "A" * 60
MHC_WITH_X_SEQ = ("A" * 180) + "X"

MHC_ALPHA_GROOVE = prepare_mhc_input(mhc_a=MHC_ALPHA_SEQ, mhc_class="I")
MHC_ALT_GROOVE = prepare_mhc_input(mhc_a=MHC_ALT_SEQ, mhc_class="I")
MHC_DR_GROOVE = prepare_mhc_input(
    mhc_a=MHC_DR_ALPHA_SEQ,
    mhc_b=MHC_DR_BETA_SEQ,
    mhc_class="II",
)
MHC_WITH_X_GROOVE = prepare_mhc_input(mhc_a=MHC_WITH_X_SEQ, mhc_class="I")


def test_load_iedb_stability_parses_multilevel_export(tmp_path):
    """IEDB-style two-row headers should parse stability rows."""
    path = tmp_path / "iedb_stability.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay", "Assay", "Assay", "MHC Restriction"],
        ["Name", "Name", "Response measured", "Units", "Quantitative measurement", "Measurement inequality", "MHC Class"],
        ["SIINFEKL", "HLA-A*02:01", "half life", "min", "360", "<", "I"],
        ["SIINFEKL", "HLA-A*02:01", "50% dissociation temperature", "C", "57", "", "I"],
        ["SIINFEKL", "HLA-A*02:01", "half maximal inhibitory concentration (IC50)", "nM", "100", "", "I"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_stability(path))
    assert len(records) == 2

    thalf = next(record for record in records if record.t_half is not None)
    assert thalf.peptide == "SIINFEKL"
    assert thalf.mhc_allele == "HLA-A*02:01"
    # Loader stores t_half in hours; 360 minutes => 6 hours.
    assert thalf.t_half == pytest.approx(6.0)
    assert thalf.t_half_qualifier == -1
    assert thalf.tm is None

    tm = next(record for record in records if record.tm is not None)
    assert tm.tm == pytest.approx(57.0)
    assert tm.t_half is None


def test_load_iedb_kinetics_parses_response_rows_and_units(tmp_path):
    """Kinetics rows encoded as response-measured entries should parse."""
    path = tmp_path / "iedb_kinetics.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay", "Assay", "Assay", "MHC Restriction"],
        ["Name", "Name", "Response measured", "Units", "Quantitative measurement", "Measurement inequality", "Class"],
        ["SIINFEKL", "HLA-A*02:01", "on rate", "nM^-1s^-1", "2.0", "", "I"],
        ["SIINFEKL", "HLA-A*02:01", "off rate", "1/s", "0.05", "<", "I"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_kinetics(path))
    assert len(records) == 2

    on_rate = next(record for record in records if record.kon is not None)
    off_rate = next(record for record in records if record.koff is not None)

    # nM^-1 s^-1 -> M^-1 s^-1
    assert on_rate.kon == pytest.approx(2.0e9)
    assert off_rate.koff == pytest.approx(0.05)
    assert off_rate.koff_qualifier == -1


def test_load_iedb_tcell_parses_multilevel_export(tmp_path):
    """IEDB two-row headers should parse T-cell rows and outcomes."""
    path = tmp_path / "iedb_tcell.csv"
    rows = [
        ["Reference", "Epitope", "Assay", "MHC Restriction", "MHC Restriction", "Epitope"],
        ["PMID", "Name", "Qualitative Measurement", "Name", "Class", "Species"],
        ["12345", "SIINFEKL", "Positive", "HLA-A*02:01", "I", "human"],
        ["12346", "GILGFVFTL", "negative-low", "HLA-B*07:02", "I", "human"],
        ["12347", "LLWNGPMAV", "0", "HLA-A2", "I", "human"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_tcell(path))
    assert len(records) == 3
    assert records[0].peptide == "SIINFEKL"
    assert records[0].response == pytest.approx(1.0)
    assert records[0].mhc_allele == "HLA-A*02:01"
    assert records[1].response == pytest.approx(0.0)
    assert records[2].response == pytest.approx(0.0)


def test_load_iedb_tcell_parses_assay_context_columns(tmp_path):
    path = tmp_path / "iedb_tcell_context.csv"
    rows = [
        [
            "Epitope",
            "MHC Restriction",
            "Assay",
            "Assay",
            "Assay",
            "Effector Cell",
            "Antigen Presenting Cell",
            "Antigen Presenting Cell",
            "In vitro Process",
            "in vitro Responder Cell",
            "in vitro Stimulator Cell",
            "MHC Restriction",
        ],
        [
            "Name",
            "Name",
            "Qualitative Measurement",
            "Response measured",
            "Method",
            "Culture Condition",
            "Name",
            "Culture Condition",
            "Process Type",
            "Name",
            "Name",
            "Class",
        ],
        [
            "SIINFEKL",
            "HLA-A*02:01",
            "Positive",
            "IFNg release",
            "ELISPOT",
            "Direct Ex Vivo",
            "dendritic cell",
            "Direct Ex Vivo",
            "Primary induction in vitro",
            "PBMC",
            "T2 cell (B cell)",
            "I",
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_tcell(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.assay_type == "IFNg release"
    assert rec.assay_method == "ELISPOT"
    assert rec.effector_culture_condition == "Direct Ex Vivo"
    assert rec.apc_name == "dendritic cell"
    assert rec.apc_culture_condition == "Direct Ex Vivo"
    assert rec.in_vitro_process_type == "Primary induction in vitro"
    assert rec.in_vitro_responder_cell == "PBMC"
    assert rec.in_vitro_stimulator_cell == "T2 cell (B cell)"


def test_load_iedb_tcell_preserves_multiple_restriction_alleles(tmp_path):
    path = tmp_path / "iedb_tcell_multi_allele.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Method"],
        ["Name", "Name", "Qualitative Measurement", "Assay Method"],
        ["ACDEFGHIKLMN", "HLA-A*02:01/HLA-DRB1*04:01", "Positive", "ELISPOT"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_tcell(path))
    assert len(records) == 1
    assert records[0].mhc_allele == "HLA-A*02:01"
    assert records[0].alleles == ["HLA-A*02:01", "HLA-DRB1*04:01"]


def test_load_iedb_binding_parses_multilevel_export(tmp_path):
    path = tmp_path / "iedb_binding.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay", "Assay", "Assay", "MHC Restriction", "Epitope"],
        ["Name", "Name", "Response measured", "Units", "Measurement inequality", "Quantitative measurement", "Class", "Species"],
        ["SIINFEKL", "HLA-A*02:01", "half maximal inhibitory concentration (IC50)", "nM", "<", "250", "I", "human"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_binding(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.peptide == "SIINFEKL"
    assert rec.mhc_allele == "HLA-A*02:01"
    assert rec.value == pytest.approx(250.0)
    assert rec.qualifier == -1
    assert rec.mhc_class == "I"
    assert rec.species == "human"


def test_load_iedb_binding_preserves_measurement_type_for_split_targets(tmp_path):
    path = tmp_path / "iedb_binding_types.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay", "Assay", "MHC Restriction"],
        ["Name", "Name", "Response measured", "Units", "Quantitative measurement", "Class"],
        ["SIINFEKL", "HLA-A*02:01", "dissociation constant KD", "nM", "50", "I"],
        ["GILGFVFTL", "HLA-A*02:01", "half maximal inhibitory concentration (IC50)", "nM", "500", "I"],
        ["LLWNGPMAV", "HLA-A*02:01", "half maximal effective concentration (EC50)", "nM", "5000", "I"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_binding(path))
    assert len(records) == 3
    assert records[0].measurement_type.lower().startswith("dissociation constant")
    assert "ic50" in records[1].measurement_type.lower()
    assert "ec50" in records[2].measurement_type.lower()

    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={"HLA-A*02:01": "A" * 181},
    )
    batch = PrestoCollator()([dataset[i] for i in range(len(dataset))])
    assert batch.target_masks["binding_kd"].tolist() == [1.0, 0.0, 0.0]
    assert batch.target_masks["binding_ic50"].tolist() == [0.0, 1.0, 0.0]
    assert batch.target_masks["binding_ec50"].tolist() == [0.0, 0.0, 1.0]


def test_record_defaults_leave_mhc_class_unknown():
    assert BindingRecord(peptide="SIINFEKL", mhc_allele="HLA-A*02:01", value=50.0).mhc_class is None
    assert KineticsRecord(peptide="SIINFEKL", mhc_allele="HLA-A*02:01").mhc_class is None
    assert StabilityRecord(peptide="SIINFEKL", mhc_allele="HLA-A*02:01").mhc_class is None
    assert ProcessingRecord(peptide="SIINFEKL").mhc_class is None
    assert ElutionRecord(peptide="SIINFEKL", alleles=["HLA-A*02:01"]).mhc_class is None
    assert TCellRecord(peptide="SIINFEKL", mhc_allele="HLA-A*02:01", response=1.0).mhc_class is None
    assert VDJdbRecord(peptide="SIINFEKL", mhc_a="HLA-A*02:01").mhc_class is None


def test_load_iedb_processing_leaves_class_none_when_absent(tmp_path):
    path = tmp_path / "iedb_processing_unknown_class.csv"
    rows = [
        ["Peptide", "Outcome"],
        ["SIINFEKL", "positive"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_processing(path))
    assert len(records) == 1
    assert records[0].mhc_class is None


def test_load_iedb_binding_infers_class_without_source_column(tmp_path):
    path = tmp_path / "iedb_binding_infer_class.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay"],
        ["Name", "Name", "Response measured", "Quantitative measurement"],
        ["PKYVKQNTLKLAT", "HLA-DRB1*01:01", "dissociation constant KD", "75"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_binding(path))
    assert len(records) == 1
    assert records[0].mhc_class == "II"


def test_load_iedb_elution_parses_multilevel_export(tmp_path):
    path = tmp_path / "iedb_elution.csv"
    rows = [
        ["Epitope", "MHC Restriction", "Assay", "Assay", "MHC Restriction"],
        ["Name", "Name", "Response measured", "Qualitative Measurement", "Class"],
        ["GILGFVFTL", "HLA-A*02:01", "ligand presentation", "Positive", "I"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_elution(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.peptide == "GILGFVFTL"
    assert rec.alleles == ["HLA-A*02:01"]
    assert rec.detected is True
    assert rec.mhc_class == "I"


def test_load_iedb_processing_skips_second_header_row(tmp_path):
    path = tmp_path / "iedb_processing.csv"
    rows = [
        ["Epitope", "Assay", "MHC Restriction"],
        ["Name", "Response measured", "Name"],
        ["SIINFEKL", "processing", "HLA-A*02:01"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_processing(path))
    assert len(records) == 1
    assert records[0].peptide == "SIINFEKL"
    assert records[0].mhc_allele == "HLA-A*02:01"


def test_load_iedb_bcell_parses_chain_types(tmp_path):
    path = tmp_path / "iedb_bcell.csv"
    rows = [
        ["Epitope", "Assay", "Assay", "Assay Antibody", "Assay Antibody", "Epitope"],
        ["Name", "Response measured", "Qualitative Measure", "Heavy chain isotype", "Light chain isotype", "Species"],
        ["SIINFEKL", "qualitative binding", "Positive", "IgG1", "kappa", "human"],
        ["GILGFVFTL", "qualitative binding", "negative", "IgM", "lambda", "human"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_iedb_bcell(path))
    assert len(records) == 2
    assert records[0].peptide == "SIINFEKL"
    assert records[0].response == pytest.approx(1.0)
    assert records[0].heavy_chain_type == "IGH"
    assert records[0].light_chain_type == "IGK"
    assert records[1].response == pytest.approx(0.0)
    assert records[1].heavy_chain_type == "IGH"
    assert records[1].light_chain_type == "IGL"


def test_load_10x_vdj_parses_t_and_b_chains(tmp_path):
    path = tmp_path / "10x.csv"
    rows = [
        [
            "barcode",
            "is_cell",
            "high_confidence",
            "chain",
            "v_gene",
            "j_gene",
            "c_gene",
            "productive",
            "cdr3",
            "cdr3_nt",
        ],
        ["cell1", "true", "true", "TRA", "TRAV12-2", "TRAJ42", "TRAC", "true", "CAVRDSNYQLIW", "TGT..."],
        ["cell1", "true", "true", "TRB", "TRBV7-9", "TRBJ2-7", "TRBC1", "true", "CASSLGQGELFF", "TGT..."],
        ["cell2", "true", "true", "IGH", "IGHV3-23", "IGHJ4", "IGHM", "true", "CARDRSTGYYYY", "TGT..."],
        ["cell2", "true", "true", "IGK", "IGKV1-39", "IGKJ1", "IGKC", "true", "CQQYNSYPYTF", "TGT..."],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    records = list(load_10x_vdj(path))
    assert len(records) == 4

    tra = records[0]
    assert tra.chain_type == "TRA"
    assert tra.phenotype == "ab_T"
    assert tra.productive is True

    igh = next(rec for rec in records if rec.chain_type == "IGH")
    assert igh.phenotype == "B_cell"
    assert igh.cdr3 == "CARDRSTGYYYY"


def test_load_vdjdb_parses_evidence_metadata(tmp_path):
    path = tmp_path / "vdjdb.txt"
    rows = [
        [
            "complex.id",
            "gene",
            "cdr3",
            "v.segm",
            "j.segm",
            "species",
            "mhc.a",
            "mhc.b",
            "mhc.class",
            "antigen.epitope",
            "antigen.species",
            "reference.id",
            "vdjdb.score",
            "method",
        ],
        [
            "1",
            "TRB",
            "CASSIRSSYEQYF",
            "TRBV7-9",
            "TRBJ2-7",
            "HomoSapiens",
            "HLA-A*02:01",
            "B2M",
            "MHCI",
            "GILGFVFTL",
            "Influenza A virus",
            "PMID:28629751",
            "3",
            '{"identification":"tetramer-sort ","singlecell":"yes","sequencing":"amplicon-seq","verification":"antigen-loaded-targets,antigen-expressing-targets"}',
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

    records = list(load_vdjdb(path))
    assert len(records) == 1
    rec = records[0]
    assert rec.peptide == "GILGFVFTL"
    assert rec.cdr3_beta == "CASSIRSSYEQYF"
    assert rec.method_identification == "tetramer-sort"
    assert rec.method_singlecell == "yes"
    assert rec.method_sequencing == "amplicon-seq"
    assert rec.method_verification == "antigen-loaded-targets,antigen-expressing-targets"
    assert rec.score == 3
    assert rec.reference_id == "PMID:28629751"


def test_load_vdjdb_tcr_evidence_derives_method_bins(tmp_path):
    path = tmp_path / "vdjdb.txt"
    rows = [
        [
            "gene",
            "cdr3",
            "species",
            "mhc.a",
            "mhc.b",
            "mhc.class",
            "antigen.epitope",
            "antigen.species",
            "reference.id",
            "vdjdb.score",
            "method",
        ],
        [
            "TRB",
            "CASSIRSSYEQYF",
            "HomoSapiens",
            "HLA-A*02:01",
            "B2M",
            "MHCI",
            "GILGFVFTL",
            "Influenza A virus",
            "PMID:28629751",
            "2",
            '{"identification":"tetramer-sort","verification":"antigen-loaded-targets","sequencing":"amplicon-seq"}',
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

    evidence = list(load_vdjdb_tcr_evidence(path))
    assert len(evidence) == 1
    rec = evidence[0]
    assert rec.method_bins == ("multimer_binding", "target_cell_functional")
    assert rec.method_sequencing == "amplicon-seq"
    assert rec.score == 2


def test_load_mcpas_tcr_evidence_preserves_method_and_ngs(tmp_path):
    path = tmp_path / "mcpas.csv"
    rows = [
        [
            "CDR3.alpha.aa",
            "CDR3.beta.aa",
            "Species",
            "Pathology",
            "Antigen.identification.method",
            "NGS",
            "Epitope.peptide",
            "MHC",
            "PubMed.ID",
        ],
        [
            "CAVRDSNYQLIW",
            "CASSIRSSYEQYF",
            "Human",
            "Influenza",
            "tetramer",
            "yes",
            "GILGFVFTL",
            "HLA-A*02:01",
            "12345",
        ],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    evidence = list(load_mcpas_tcr_evidence(path))
    assert len(evidence) == 1
    rec = evidence[0]
    assert rec.peptide == "GILGFVFTL"
    assert rec.method_identification == "tetramer"
    assert rec.method_sequencing == "yes"
    assert rec.reference_id == "12345"


def test_presto_dataset_defaults_class_i_to_groove_halves():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={"HLA-A*02:01": MHC_ALPHA_SEQ},
    )

    sample = dataset[0]
    assert sample.mhc_a == MHC_ALPHA_GROOVE.groove_half_1
    assert sample.mhc_b == MHC_ALPHA_GROOVE.groove_half_2


def test_presto_dataset_raises_on_unresolved_mhc_allele_in_strict_mode():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*99:99",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    with pytest.raises(ValueError, match="Unresolved MHC allele without sequence"):
        PrestoDataset(
            binding_records=records,
            mhc_sequences={},
            strict_mhc_resolution=True,
        )


def test_presto_dataset_allows_unresolved_mhc_allele_when_not_strict():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*99:99",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={},
        strict_mhc_resolution=False,
    )
    sample = dataset[0]
    assert sample.mhc_a == ""


def test_presto_dataset_keeps_class_i_no_mhc_beta_negative_empty_beta_chain():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=80000.0,
            mhc_class="I",
            species="human",
            source="synthetic_negative_no_mhc_beta",
        )
    ]
    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={"HLA-A*02:01": MHC_ALPHA_SEQ},
    )

    sample = dataset[0]
    assert sample.mhc_b == ""


def test_presto_dataset_keeps_no_mhc_alpha_negative_empty_alpha_chain():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=80000.0,
            mhc_class="I",
            species="human",
            source="synthetic_negative_no_mhc_alpha",
        )
    ]
    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={"HLA-A*02:01": MHC_ALPHA_SEQ},
    )

    sample = dataset[0]
    assert sample.mhc_a == ""


def test_presto_dataset_class_ii_dr_beta_only_auto_pairs_default_dra():
    records = [
        BindingRecord(
            peptide="KPVSKMRMATPLLMQALPM",
            mhc_allele="HLA-DRB1*04:01",
            value=500.0,
            mhc_class="II",
            species="human",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        binding_records=records,
        mhc_sequences={
            "HLA-DRA*01:01": MHC_DR_ALPHA_SEQ,
            "HLA-DRB1*04:01": MHC_DR_BETA_SEQ,
        },
    )

    sample = dataset[0]
    assert sample.mhc_a == MHC_DR_GROOVE.groove_half_1
    assert sample.mhc_b == MHC_DR_GROOVE.groove_half_2


def test_presto_dataset_elution_preserves_multi_allele_bag_instances():
    records = [
        ElutionRecord(
            peptide="SIINFEKL",
            alleles=["HLA-A*02:01", "HLA-B*07:02"],
            detected=True,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        elution_records=records,
        mhc_sequences={
            "HLA-A*02:01": MHC_ALPHA_SEQ,
            "HLA-B*07:02": MHC_ALT_SEQ,
        },
    )

    sample = dataset[0]
    assert sample.elution_label == 1.0
    assert sample.mil_mhc_a_list == [
        MHC_ALPHA_GROOVE.groove_half_1,
        MHC_ALT_GROOVE.groove_half_1,
    ]
    assert sample.mil_mhc_b_list is not None
    assert len(sample.mil_mhc_b_list) == 2
    assert sample.mil_mhc_b_list == [
        MHC_ALPHA_GROOVE.groove_half_2,
        MHC_ALT_GROOVE.groove_half_2,
    ]
    assert sample.mil_mhc_class_list == ["I", "I"]
    assert sample.mil_species_list == ["human", "human"]


def test_presto_dataset_tcell_builds_pathway_mil_for_mixed_class_ambiguous_assay():
    records = [
        TCellRecord(
            peptide="ACDEFGHIKLMN",
            mhc_allele="HLA-A*02:01",
            alleles=["HLA-A*02:01", "HLA-DRB1*04:01"],
            response=1.0,
            mhc_class=None,
            assay_method="ELISPOT",
            assay_type="IFNg release",
            effector_culture_condition="Direct Ex Vivo",
            in_vitro_responder_cell="PBMC",
            species="human",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        tcell_records=records,
        mhc_sequences={
            "HLA-A*02:01": MHC_ALPHA_SEQ,
            "HLA-DRA*01:01": MHC_DR_ALPHA_SEQ,
            "HLA-DRB1*04:01": MHC_DR_BETA_SEQ,
        },
    )

    sample = dataset[0]
    assert sample.use_tcell_pathway_mil is True
    assert sample.tcell_mil_mhc_class_list == ["I", "II"]
    assert sample.tcell_mil_mhc_a_list == [
        MHC_ALPHA_GROOVE.groove_half_1,
        MHC_DR_GROOVE.groove_half_1,
    ]
    assert sample.tcell_mil_mhc_b_list == [
        MHC_ALPHA_GROOVE.groove_half_2,
        MHC_DR_GROOVE.groove_half_2,
    ]
    assert sample.tcell_mil_species_list == ["human", "human"]


def test_presto_dataset_vdjdb_resolves_class_ii_beta_allele_partner():
    records = [
        VDJdbRecord(
            cdr3_alpha="CAVRDTNTGKLIF",
            cdr3_beta="CASSIRSSYEQYF",
            peptide="AGFKGEQGPKGEPG",
            mhc_a="HLA-DRA*01:01",
            mhc_b="HLA-DRB1*01:01",
            gene="unknown",
            species="human",
            antigen_species="human",
        )
    ]
    dataset = PrestoDataset(
        vdjdb_records=records,
        mhc_sequences={
            "HLA-DRA*01:01": MHC_DR_ALPHA_SEQ,
            "HLA-DRB1*01:01": MHC_DR_BETA_SEQ,
        },
    )

    sample = dataset[0]
    assert sample.mhc_a == MHC_DR_GROOVE.groove_half_1
    assert sample.mhc_b == MHC_DR_GROOVE.groove_half_2


def test_presto_dataset_rejects_noncanonical_mhc_residues():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    with pytest.raises(ValueError, match="Non-canonical residue"):
        PrestoDataset(
            binding_records=records,
            mhc_sequences={"HLA-A*02:01": ("A" * 150) + "Z"},
        )


def test_presto_dataset_rejects_short_mhc_alpha_chain():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    with pytest.raises(ValueError, match="minimum accepted groove-bearing fragment"):
        PrestoDataset(
            binding_records=records,
            mhc_sequences={"HLA-A*02:01": MHC_SHORT_SEQ},
        )


def test_presto_dataset_warns_on_ambiguous_x_in_mhc_sequence():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    with pytest.warns(RuntimeWarning, match="Detected ambiguous residue 'X'"):
        dataset = PrestoDataset(
            binding_records=records,
            mhc_sequences={"HLA-A*02:01": MHC_WITH_X_SEQ},
        )
    assert dataset[0].mhc_a == MHC_WITH_X_GROOVE.groove_half_1
    assert dataset[0].mhc_b == MHC_WITH_X_GROOVE.groove_half_2


def test_presto_dataset_rejects_nucleotide_like_mhc_chain():
    records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        )
    ]
    with pytest.raises(ValueError, match="Likely nucleotide sequence loaded for MHC chain"):
        PrestoDataset(
            binding_records=records,
            mhc_sequences={"HLA-A*02:01": "ACG" * 60},
        )


def test_presto_dataset_ignores_10x_chain_supervision_samples():
    sc10x_records = [
        Sc10xVDJRecord(
            barcode="cell1",
            chain_type="TRB",
            cdr3="CASSLGQGELFF",
            phenotype="ab_T",
            species="human",
            source="10x",
        )
    ]
    with pytest.warns(RuntimeWarning, match="Ignoring sc10x_records"):
        dataset = PrestoDataset(sc10x_records=sc10x_records)
    assert len(dataset) == 0


def test_presto_dataset_converts_vdjdb_to_tcr_evidence_samples():
    dataset = PrestoDataset(
        vdjdb_records=[
            VDJdbRecord(
                peptide="SIINFEKL",
                mhc_a="HLA-A*02:01",
                mhc_class="I",
                species="human",
                cdr3_beta="CASSIRSSYEQYF",
                method_identification="tetramer",
                method_verification="IFNg release",
                source="vdjdb",
            )
        ],
        mhc_sequences={"HLA-A*02:01": "A" * 181},
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.tcr_evidence_label == pytest.approx(1.0)
    assert sample.tcr_evidence_method_bins == (
        "multimer_binding",
        "functional_readout",
    )
    assert sample.assay_group == "tcr_evidence"


def test_create_dataloader_balanced_batches_mix_tasks():
    binding_records = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            source="iedb",
        ),
        BindingRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            value=70000.0,
            mhc_class="I",
            source="synthetic_negative",
        ),
    ]
    elution_records = [
        ElutionRecord(
            peptide="SIINFEKL",
            alleles=["HLA-A*02:01"],
            detected=True,
            mhc_class="I",
            source="iedb",
        ),
        ElutionRecord(
            peptide="NLVPMVATV",
            alleles=["HLA-A*02:01"],
            detected=False,
            mhc_class="I",
            source="synthetic_negative",
        ),
    ]
    tcell_records = [
        TCellRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            response=1.0,
            mhc_class="I",
            source="iedb",
        ),
        TCellRecord(
            peptide="NLVPMVATV",
            mhc_allele="HLA-A*02:01",
            response=0.0,
            mhc_class="I",
            source="synthetic_negative_from_binding",
        ),
    ]
    dataset = PrestoDataset(
        binding_records=binding_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        mhc_sequences={"HLA-A*02:01": "A" * 181},
    )
    loader = create_dataloader(
        dataset,
        batch_size=9,
        shuffle=True,
        collator=PrestoCollator(),
        balanced=True,
        seed=13,
    )
    batch = next(iter(loader))

    prefixes = {sample_id.split("_", 1)[0] for sample_id in batch.sample_ids}
    assert "bind" in prefixes
    assert "elut" in prefixes
    assert "tcell" in prefixes


def test_create_dataloader_balanced_batches_use_proportional_task_quotas():
    binding_records = [
        BindingRecord(
            peptide=f"SIINFEK{i % 10}",
            mhc_allele="HLA-A*02:01",
            value=50.0 + float(i),
            measurement_type="IC50",
            mhc_class="I",
            source="iedb",
        )
        for i in range(100)
    ]
    tcell_records = [
        TCellRecord(
            peptide="NLVPMVATV",
            mhc_allele="HLA-A*02:01",
            response=1.0,
            mhc_class="I",
            source="iedb",
        )
    ]
    dataset = PrestoDataset(
        binding_records=binding_records,
        tcell_records=tcell_records,
        mhc_sequences={"HLA-A*02:01": MHC_ALPHA_SEQ},
    )
    loader = create_dataloader(
        dataset,
        batch_size=20,
        shuffle=True,
        balanced=True,
        seed=7,
    )

    batch_indices = next(iter(loader.batch_sampler))
    task_counts = Counter(dataset[idx].assay_group for idx in batch_indices)
    assert task_counts["tcell_response"] == 1
    assert task_counts["binding_ic50"] == 19
