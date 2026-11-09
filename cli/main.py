"""Presto CLI main entry point.

Usage:
    presto data download [--all] [--source SOURCE] [--agree-iedb-terms]
    presto data list [--local] [--source SOURCE]
    presto data process [--dataset DATASET] [--filter FILTER]
    presto weights list [--registry PATH_OR_URL]
    presto weights download --name MODEL [--registry PATH_OR_URL]
    presto train --observations-manifest manifest.json [--preflight]
    presto predict presentation [options]
    presto predict tile [options]
    presto predict recognition [options]
    presto predict chain [options]
    presto evaluate synthetic [options]
"""

import argparse
import sys
from typing import List, Optional

from .. import __version__
from .data import (
    cmd_data_download,
    cmd_data_list,
    cmd_data_process,
    cmd_data_info,
    cmd_data_dedup,
    cmd_data_merge,
    cmd_data_mhc_index_build,
    cmd_data_mhc_index_augment,
    cmd_data_mhc_index_refresh,
    cmd_data_mhc_index_mouse_overlay,
    cmd_data_mhc_index_report,
    cmd_data_mhc_index_validate,
    cmd_data_mhc_index_resolve,
)
from .train import cmd_train
from .predict import (
    cmd_predict_presentation,
    cmd_predict_tile,
    cmd_predict_recognition,
)
from .evaluate import cmd_evaluate_synthetic
from .weights import cmd_weights_download, cmd_weights_list


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="presto",
        description="Presto - Peptide-Receptor Embedding for Shared T-cell Ontology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"presto {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ==========================================================================
    # data subcommand
    # ==========================================================================
    data_parser = subparsers.add_parser(
        "data",
        help="Data management: download, list, and process datasets",
    )
    data_subparsers = data_parser.add_subparsers(
        dest="data_command",
        help="Data management commands",
    )

    # data download
    download_parser = data_subparsers.add_parser(
        "download",
        help="Download datasets from online sources (IEDB, VDJdb, etc.)",
    )
    download_parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded data (default: ./data)",
    )
    download_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        action="append",
        help="Specific dataset(s) to download. Can be repeated. "
        "Use 'presto data list' to see available datasets.",
    )
    download_parser.add_argument(
        "--source",
        "-s",
        type=str,
        action="append",
        choices=["iedb", "vdjdb", "mcpas", "imgt", "ipd_mhc", "10x", "pird", "stcrdab"],
        help="Download all datasets from specific source(s)",
    )
    download_parser.add_argument(
        "--category",
        "-c",
        type=str,
        action="append",
        choices=["binding", "tcell", "bcell", "tcr", "mhc_sequence", "elution", "vdj_genes"],
        help="Download all datasets of specific category",
    )
    download_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Download all available datasets",
    )
    download_parser.add_argument(
        "--agree-iedb-terms",
        action="store_true",
        help="Agree to IEDB terms of use (required for IEDB downloads). "
        "See: https://www.iedb.org/terms_of_use.php",
    )
    download_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if files exist",
    )
    download_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    download_parser.set_defaults(func=cmd_data_download)

    # data list
    list_parser = data_subparsers.add_parser(
        "list",
        help="List available or downloaded datasets",
    )
    list_parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        help="List locally downloaded datasets instead of available ones",
    )
    list_parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory to check for local datasets (default: ./data)",
    )
    list_parser.add_argument(
        "--source",
        "-s",
        type=str,
        action="append",
        choices=["iedb", "vdjdb", "mcpas", "imgt", "ipd_mhc", "10x", "pird", "stcrdab"],
        help="Filter by source",
    )
    list_parser.add_argument(
        "--category",
        "-c",
        type=str,
        action="append",
        choices=["binding", "tcell", "bcell", "tcr", "mhc_sequence", "elution", "vdj_genes"],
        help="Filter by category",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    list_parser.set_defaults(func=cmd_data_list)

    # data info
    info_parser = data_subparsers.add_parser(
        "info",
        help="Show detailed information about a dataset",
    )
    info_parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name to show info for",
    )
    info_parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    info_parser.set_defaults(func=cmd_data_info)

    # data process
    process_parser = data_subparsers.add_parser(
        "process",
        help="Process/filter downloaded datasets for training",
    )
    process_parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory with downloaded datasets (default: ./data)",
    )
    process_parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        default="./data/processed",
        help="Output directory for processed data (default: ./data/processed)",
    )
    process_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        action="append",
        help="Specific dataset(s) to process",
    )
    process_parser.add_argument(
        "--species",
        type=str,
        default="human",
        help="Filter by species (default: human)",
    )
    process_parser.add_argument(
        "--mhc-class",
        type=str,
        choices=["I", "II", "all"],
        default="all",
        help="Filter by MHC class (default: all)",
    )
    process_parser.add_argument(
        "--min-peptide-length",
        type=int,
        default=8,
        help="Minimum peptide length (default: 8)",
    )
    process_parser.add_argument(
        "--max-peptide-length",
        type=int,
        default=25,
        help="Maximum peptide length (default: 25)",
    )
    process_parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate peptide-MHC pairs",
    )
    process_parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train/val/test (e.g., --split 0.8 0.1 0.1)",
    )
    process_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    process_parser.set_defaults(func=cmd_data_process)

    # data dedup
    dedup_parser = data_subparsers.add_parser(
        "dedup",
        help="Deduplicate assay data by reference (PubMed ID / DOI)",
    )
    dedup_parser.add_argument(
        "input",
        type=str,
        help="Input file (CSV or TSV)",
    )
    dedup_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: input_deduped.tsv)",
    )
    dedup_parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["binding", "tcell"],
        default="binding",
        help="Data type for deduplication strategy (default: binding)",
    )
    dedup_parser.add_argument(
        "--aggregate",
        type=str,
        choices=["median", "mean", "first", "best"],
        default="median",
        help="How to aggregate multiple values from same reference (default: median)",
    )
    dedup_parser.add_argument(
        "--prefer-recent",
        action="store_true",
        default=True,
        help="Prefer more recent publications (default: true)",
    )
    dedup_parser.add_argument(
        "--no-prefer-recent",
        action="store_false",
        dest="prefer_recent",
        help="Don't prefer recent publications",
    )
    dedup_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    dedup_parser.set_defaults(func=cmd_data_dedup)

    # data merge (cross-source deduplication)
    merge_parser = data_subparsers.add_parser(
        "merge",
        help="Merge and deduplicate data across sources (IEDB, VDJdb, McPAS, etc.)",
    )
    merge_parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory with downloaded datasets (default: ./data)",
    )
    merge_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: datadir/merged_deduped.tsv)",
    )
    merge_parser.add_argument(
        "--assay-outdir",
        type=str,
        help=(
            "Output directory for simplified one-file-per-assay CSV exports "
            "(default: datadir/merged_assays)"
        ),
    )
    merge_parser.add_argument(
        "--per-assay-csv",
        action="store_true",
        help="Write one simplified CSV per assay bucket (default: disabled)",
    )
    merge_parser.add_argument(
        "--types",
        "-t",
        type=str,
        nargs="+",
        choices=["binding", "tcell", "bcell", "tcr"],
        help="Record types to include (default: all)",
    )
    merge_parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics in JSON format",
    )
    merge_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    merge_parser.set_defaults(func=cmd_data_merge)

    # data mhc-index (build + resolve)
    mhc_index_parser = data_subparsers.add_parser(
        "mhc-index",
        help="Build and query an MHC allele index",
    )
    mhc_index_subparsers = mhc_index_parser.add_subparsers(
        dest="mhc_index_command",
        help="MHC index commands",
        required=True,
    )

    mhc_refresh = mhc_index_subparsers.add_parser(
        "refresh",
        help="Resolve IMGT/IPD-MHC inputs and rebuild the MHC index",
    )
    mhc_refresh.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory for downloaded datasets (default: ./data)",
    )
    mhc_refresh.add_argument(
        "--imgt-fasta",
        type=str,
        help="Optional override path for IMGT/HLA FASTA",
    )
    mhc_refresh.add_argument(
        "--ipd-mhc-dir",
        type=str,
        help="Optional override path for IPD-MHC FASTA or directory",
    )
    mhc_refresh.add_argument(
        "--out-csv",
        type=str,
        help="Output CSV path (default: datadir/mhc_index.csv)",
    )
    mhc_refresh.add_argument(
        "--out-fasta",
        type=str,
        help="Optional output FASTA path",
    )
    mhc_refresh.add_argument(
        "--download-missing",
        action="store_true",
        help="Download IMGT/IPD-MHC datasets when missing",
    )
    mhc_refresh.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    mhc_refresh.set_defaults(func=cmd_data_mhc_index_refresh)

    mhc_mouse_overlay = mhc_index_subparsers.add_parser(
        "mouse-overlay",
        help="Build mouse MHC allele overlay FASTA/CSV from IMGT nomenclature + UniProt",
    )
    mhc_mouse_overlay.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory for overlay outputs (default: ./data)",
    )
    mhc_mouse_overlay.add_argument(
        "--out-csv",
        type=str,
        help="Output catalog CSV path (default: datadir/ipd_mhc/mouse_uniprot_overlay.csv)",
    )
    mhc_mouse_overlay.add_argument(
        "--out-fasta",
        type=str,
        help="Output FASTA path (default: datadir/ipd_mhc/mouse_uniprot_overlay.fasta)",
    )
    mhc_mouse_overlay.add_argument(
        "--imgt-url",
        type=str,
        help="Override IMGT mouse nomenclature URL",
    )
    mhc_mouse_overlay.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include unreviewed UniProt entries (default: reviewed only)",
    )
    mhc_mouse_overlay.add_argument(
        "--max-genes",
        type=int,
        default=0,
        help="Optional cap on number of IMGT genes processed (0 means all)",
    )
    mhc_mouse_overlay.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    mhc_mouse_overlay.set_defaults(func=cmd_data_mhc_index_mouse_overlay)

    mhc_build = mhc_index_subparsers.add_parser(
        "build",
        help="Build MHC index CSV (and optional FASTA) from IMGT/IPD-MHC",
    )
    mhc_build.add_argument(
        "--imgt-fasta",
        type=str,
        help="Path to IMGT/HLA protein FASTA (e.g., hla_prot.fasta)",
    )
    mhc_build.add_argument(
        "--ipd-mhc-dir",
        type=str,
        help="Path to IPD-MHC directory or FASTA file",
    )
    mhc_build.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path for the index",
    )
    mhc_build.add_argument(
        "--out-fasta",
        type=str,
        help="Optional FASTA output path (index sequences)",
    )
    mhc_build.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    mhc_build.set_defaults(func=cmd_data_mhc_index_build)

    mhc_augment = mhc_index_subparsers.add_parser(
        "augment",
        help="Augment an existing MHC index with groove-centric columns",
    )
    mhc_augment.add_argument(
        "--index-csv",
        type=str,
        required=True,
        help="Path to the built MHC index CSV",
    )
    mhc_augment.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path for the augmented index",
    )
    mhc_augment.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    mhc_augment.set_defaults(func=cmd_data_mhc_index_augment)

    mhc_report = mhc_index_subparsers.add_parser(
        "report",
        help="Summarize counts from a built MHC index CSV",
    )
    mhc_report.add_argument(
        "--index-csv",
        type=str,
        required=True,
        help="Path to the built MHC index CSV",
    )
    mhc_report.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    mhc_report.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path (default: stdout)",
    )
    mhc_report.set_defaults(func=cmd_data_mhc_index_report)

    mhc_validate = mhc_index_subparsers.add_parser(
        "validate",
        help="Validate MHC index schema and canonical normalization",
    )
    mhc_validate.add_argument(
        "--index-csv",
        type=str,
        required=True,
        help="Path to the built MHC index CSV",
    )
    mhc_validate.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    mhc_validate.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path (default: stdout)",
    )
    mhc_validate.set_defaults(func=cmd_data_mhc_index_validate)

    mhc_resolve = mhc_index_subparsers.add_parser(
        "resolve",
        help="Resolve allele names using a built MHC index CSV",
    )
    mhc_resolve.add_argument(
        "--index-csv",
        type=str,
        required=True,
        help="Path to the built MHC index CSV",
    )
    mhc_resolve.add_argument(
        "--alleles",
        type=str,
        help="Comma-separated allele list",
    )
    mhc_resolve.add_argument(
        "--allele-file",
        type=str,
        help="File with alleles (one per line or CSV/TSV with header)",
    )
    mhc_resolve.add_argument(
        "--column",
        type=str,
        help="Column name to read from CSV/TSV (default: allele)",
    )
    mhc_resolve.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    mhc_resolve.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path (default: stdout)",
    )
    mhc_resolve.add_argument(
        "--no-seq",
        action="store_true",
        help="Omit sequences from output",
    )
    mhc_resolve.set_defaults(func=cmd_data_mhc_index_resolve)

    # ==========================================================================
    # weights subcommand
    # ==========================================================================
    weights_parser = subparsers.add_parser(
        "weights",
        help="List and download pretrained model checkpoints",
    )
    weights_subparsers = weights_parser.add_subparsers(
        dest="weights_command",
        help="Weights commands",
    )

    weights_list = weights_subparsers.add_parser(
        "list",
        help="List available weight artifacts from a registry",
    )
    weights_list.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Registry JSON file path or URL (default: built-in registry)",
    )
    weights_list.add_argument(
        "--json",
        action="store_true",
        help="Output JSON",
    )
    weights_list.set_defaults(func=cmd_weights_list)

    weights_download = weights_subparsers.add_parser(
        "download",
        help="Download model weights by registry name or direct URL",
    )
    weights_download.add_argument(
        "--name",
        type=str,
        default=None,
        help="Registry model name",
    )
    weights_download.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct checkpoint URL (bypasses registry lookup)",
    )
    weights_download.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Registry JSON file path or URL",
    )
    weights_download.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output checkpoint path (default: ~/.cache/presto/weights/...)",
    )
    weights_download.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached weights when --output is not provided",
    )
    weights_download.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if output file exists",
    )
    weights_download.set_defaults(func=cmd_weights_download)

    # ==========================================================================
    # train subcommand
    # ==========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train the shared Presto model from an observation manifest",
    )
    train_parser.add_argument(
        "train_command",
        nargs="?",
        choices=("unified", "synthetic", "pretrain", "finetune"),
        help="Compatibility aliases; every mode uses the same manifest trainer",
    )
    from .training_sources import add_training_source_arguments

    add_training_source_arguments(train_parser)
    train_parser.set_defaults(func=cmd_train)

    # ==========================================================================
    # predict subcommand
    # ==========================================================================
    predict_parser = subparsers.add_parser(
        "predict",
        help="Model inference (presentation, recognition, or chain classification)",
    )
    predict_subparsers = predict_parser.add_subparsers(
        dest="predict_command",
        help="Prediction modes",
    )

    predict_presentation = predict_subparsers.add_parser(
        "presentation",
        help="Predict processing/binding/presentation probabilities",
    )
    predict_presentation.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    predict_presentation.add_argument("--peptide", type=str, required=True, help="Peptide sequence")
    predict_presentation.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_presentation.add_argument(
        "--mhc-sequence", type=str, default=None, help="MHC alpha sequence"
    )
    predict_presentation.add_argument(
        "--mhc-b-sequence", type=str, default=None, help="MHC beta sequence"
    )
    predict_presentation.add_argument(
        "--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class"
    )
    predict_presentation.add_argument(
        "--species", type=str, default=None, help="Species label for class-I beta2m resolution"
    )
    predict_presentation.add_argument(
        "--mhc-species", type=str, default=None, help="Override MHC species latent/probability path"
    )
    predict_presentation.add_argument(
        "--immune-species",
        type=str,
        default=None,
        help="Override immune-system species conditioning",
    )
    predict_presentation.add_argument(
        "--species-of-origin",
        type=str,
        default=None,
        help="Override peptide species-of-origin latent",
    )
    predict_presentation.add_argument("--flank-n", type=str, default=None, help="N-terminal flank")
    predict_presentation.add_argument("--flank-c", type=str, default=None, help="C-terminal flank")
    predict_presentation.add_argument(
        "--flank-n-is-terminus",
        action="store_true",
        help="The supplied N-flank reaches an observed protein boundary",
    )
    predict_presentation.add_argument(
        "--flank-c-is-terminus",
        action="store_true",
        help="The supplied C-flank reaches an observed protein boundary",
    )
    predict_presentation.add_argument(
        "--index-csv",
        type=str,
        default=None,
        help="Optional built MHC index CSV for allele sequence lookup",
    )
    predict_presentation.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_presentation.add_argument(
        "--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory"
    )
    predict_presentation.add_argument(
        "--d-model", dest="d_model", type=int, default=None, help="Model dimension"
    )
    predict_presentation.add_argument(
        "--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers"
    )
    predict_presentation.add_argument(
        "--n-heads", dest="n_heads", type=int, default=None, help="Attention heads"
    )
    predict_presentation.add_argument("--device", type=str, default=None, help="Device")
    predict_presentation.add_argument("--json", action="store_true", help="Output JSON")
    predict_presentation.add_argument("--output", type=str, default=None, help="Output path")
    predict_presentation.set_defaults(func=cmd_predict_presentation)

    predict_tile = predict_subparsers.add_parser(
        "tile",
        help="Tile presentation predictions over all subsequences of a protein",
    )
    predict_tile.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_tile.add_argument("--protein-sequence", type=str, default=None, help="Protein sequence")
    predict_tile.add_argument(
        "--protein-file", type=str, default=None, help="Path to FASTA/plain sequence file"
    )
    predict_tile.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_tile.add_argument("--mhc-sequence", type=str, default=None, help="MHC alpha sequence")
    predict_tile.add_argument("--mhc-b-sequence", type=str, default=None, help="MHC beta sequence")
    predict_tile.add_argument(
        "--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class"
    )
    predict_tile.add_argument(
        "--species", type=str, default=None, help="Species label for class-I beta2m resolution"
    )
    predict_tile.add_argument(
        "--mhc-species", type=str, default=None, help="Override MHC species latent/probability path"
    )
    predict_tile.add_argument(
        "--immune-species",
        type=str,
        default=None,
        help="Override immune-system species conditioning",
    )
    predict_tile.add_argument(
        "--species-of-origin",
        type=str,
        default=None,
        help="Override peptide species-of-origin latent",
    )
    predict_tile.add_argument(
        "--min-length", type=int, default=8, help="Minimum peptide length for tiling"
    )
    predict_tile.add_argument(
        "--max-length", type=int, default=15, help="Maximum peptide length for tiling"
    )
    predict_tile.add_argument(
        "--flank-size", type=int, default=15, help="Context flank size on each side"
    )
    predict_tile.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for tiled inference"
    )
    predict_tile.add_argument(
        "--top-k", type=int, default=100, help="Keep top-k hits (<=0 keeps all)"
    )
    predict_tile.add_argument(
        "--sort-by",
        type=str,
        choices=["presentation", "binding", "processing"],
        default="presentation",
        help="Sort metric for reported tiled hits",
    )
    predict_tile.add_argument(
        "--index-csv",
        type=str,
        default=None,
        help="Optional built MHC index CSV for allele sequence lookup",
    )
    predict_tile.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_tile.add_argument("--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory")
    predict_tile.add_argument(
        "--d-model", dest="d_model", type=int, default=None, help="Model dimension"
    )
    predict_tile.add_argument(
        "--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers"
    )
    predict_tile.add_argument(
        "--n-heads", dest="n_heads", type=int, default=None, help="Attention heads"
    )
    predict_tile.add_argument("--device", type=str, default=None, help="Device")
    predict_tile.add_argument("--json", action="store_true", help="Output JSON")
    predict_tile.add_argument("--output", type=str, default=None, help="Output path")
    predict_tile.set_defaults(func=cmd_predict_tile)

    predict_recognition = predict_subparsers.add_parser(
        "recognition",
        help="Predict repertoire-level recognition/immunogenicity",
    )
    predict_recognition.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint"
    )
    predict_recognition.add_argument("--peptide", type=str, required=True, help="Peptide sequence")
    predict_recognition.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_recognition.add_argument(
        "--mhc-sequence", type=str, default=None, help="MHC alpha sequence"
    )
    predict_recognition.add_argument(
        "--mhc-b-sequence", type=str, default=None, help="MHC beta sequence"
    )
    predict_recognition.add_argument(
        "--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class"
    )
    predict_recognition.add_argument(
        "--species", type=str, default=None, help="Species label for class-I beta2m resolution"
    )
    predict_recognition.add_argument(
        "--mhc-species", type=str, default=None, help="Override MHC species latent/probability path"
    )
    predict_recognition.add_argument(
        "--immune-species",
        type=str,
        default=None,
        help="Override immune-system species conditioning",
    )
    predict_recognition.add_argument(
        "--species-of-origin",
        type=str,
        default=None,
        help="Override peptide species-of-origin latent",
    )
    predict_recognition.add_argument(
        "--index-csv",
        type=str,
        default=None,
        help="Optional built MHC index CSV for allele sequence lookup",
    )
    predict_recognition.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_recognition.add_argument(
        "--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory"
    )
    predict_recognition.add_argument(
        "--d-model", dest="d_model", type=int, default=None, help="Model dimension"
    )
    predict_recognition.add_argument(
        "--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers"
    )
    predict_recognition.add_argument(
        "--n-heads", dest="n_heads", type=int, default=None, help="Attention heads"
    )
    predict_recognition.add_argument("--device", type=str, default=None, help="Device")
    predict_recognition.add_argument("--json", action="store_true", help="Output JSON")
    predict_recognition.add_argument("--output", type=str, default=None, help="Output path")
    predict_recognition.set_defaults(func=cmd_predict_recognition)

    # ==========================================================================
    # evaluate subcommand
    # ==========================================================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluation workflows",
    )
    eval_subparsers = eval_parser.add_subparsers(
        dest="evaluate_command",
        help="Evaluation modes",
    )

    eval_synth = eval_subparsers.add_parser(
        "synthetic",
        help="Evaluate on synthetic data",
    )
    eval_synth.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    eval_synth.add_argument("--batch_size", type=int, default=16, help="Batch size")
    eval_synth.add_argument(
        "--d-model", dest="d_model", type=int, default=None, help="Model dimension"
    )
    eval_synth.add_argument(
        "--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers"
    )
    eval_synth.add_argument(
        "--n-heads", dest="n_heads", type=int, default=None, help="Attention heads"
    )
    eval_synth.add_argument("--n_binding", type=int, default=200, help="Binding samples")
    eval_synth.add_argument("--n_elution", type=int, default=100, help="Elution samples")
    eval_synth.add_argument("--n_tcr", type=int, default=100, help="TCR samples")
    eval_synth.add_argument("--data_dir", type=str, default=None, help="Data directory")
    eval_synth.add_argument("--seed", type=int, default=42, help="Random seed")
    eval_synth.add_argument("--device", type=str, default=None, help="Device")
    eval_synth.add_argument("--json", action="store_true", help="Output JSON")
    eval_synth.set_defaults(func=cmd_evaluate_synthetic)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    from ..training.config_io import explicit_cli_destinations

    args._explicit_cli_dests = explicit_cli_destinations(
        parser, list(sys.argv[1:] if argv is None else argv), args
    )

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "data" and args.data_command is None:
        # Print data subcommand help
        parser.parse_args(["data", "--help"])
        return 0
    if args.command == "weights" and getattr(args, "weights_command", None) is None:
        parser.parse_args(["weights", "--help"])
        return 0
    if args.command == "predict" and getattr(args, "predict_command", None) is None:
        parser.parse_args(["predict", "--help"])
        return 0
    if args.command == "evaluate" and getattr(args, "evaluate_command", None) is None:
        parser.parse_args(["evaluate", "--help"])
        return 0

    # Execute the command
    if hasattr(args, "func"):
        try:
            return args.func(args)
        except KeyboardInterrupt:
            print("\nInterrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
