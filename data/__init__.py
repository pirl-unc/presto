"""Data handling: vocab, tokenizers, loaders."""

from .vocab import (
    AA_VOCAB,
    CHAIN_TYPES,
    CHAIN_TYPES_FULL,
    CHAIN_TYPES_CDR3,
    CDR3_TO_FULL,
    FULL_TO_CDR3,
    CELL_TYPES,
    MHC_TYPES,
    SPECIES,
    VALID_CHAIN_CELL,
    CELL_MHC_COMPATIBILITY,
    is_valid_chain_cell,
    is_compatible_cell_mhc,
    is_cdr3_only,
    get_base_chain_type,
)
from .tokenizer import Tokenizer
from .collate import (
    PrestoSample,
    PrestoBatch,
    PrestoCollator,
    collate_dict_batch,
)
from .allele_resolver import (
    AlleleResolver,
    AlleleRecord,
    normalize_allele_name,
    infer_mhc_class,
    infer_gene,
    infer_species,
)
from .loaders import (
    # Record types
    BindingRecord,
    KineticsRecord,
    StabilityRecord,
    ProcessingRecord,
    ElutionRecord,
    TCellRecord,
    VDJdbRecord,
    TCRpMHCRecord,  # Backward compat alias for VDJdbRecord
    # IEDB loaders
    load_iedb_binding,
    load_iedb_kinetics,
    load_iedb_stability,
    load_iedb_processing,
    load_iedb_elution,
    load_iedb_tcell,
    # VDJdb loader
    load_vdjdb,
    # Simple loaders (backward compat)
    load_binding_csv,
    load_elution_csv,
    load_tcr_pmhc_csv,
    load_mhc_fasta,
    # Dataset and dataloader
    PrestoDataset,
    create_dataloader,
    # Synthetic data generators
    generate_synthetic_binding_data,
    generate_synthetic_kinetics_data,
    generate_synthetic_stability_data,
    generate_synthetic_processing_data,
    generate_synthetic_elution_data,
    generate_synthetic_tcell_data,
    generate_synthetic_vdjdb_data,
    generate_synthetic_tcr_data,  # Backward compat alias
    generate_synthetic_mhc_sequences,
    # Writers
    write_binding_csv,
    write_elution_csv,
    write_tcr_csv,
    write_vdjdb_tsv,
    write_mhc_fasta,
    # Allele and gene lists for synthetic data
    CLASS_I_ALLELES,
    CLASS_II_ALLELES,
    V_ALPHA_GENES,
    V_BETA_GENES,
    J_ALPHA_GENES,
    J_BETA_GENES,
    # Note: CELL_TYPES from loaders conflicts with vocab.CELL_TYPES
    # Use presto.data.loaders.CELL_TYPES for synthetic data cell types
    TISSUES,
    BINDING_ASSAYS,
    TCELL_ASSAYS,
)
from .downloaders import (
    # Dataset registry
    DATASETS,
    DatasetInfo,
    DownloadManifest,
    DownloadState,
    # Download functions
    download_dataset,
    download_all,
    list_datasets,
    list_local_datasets,
    get_dataset_path,
)
