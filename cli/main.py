"""Presto CLI main entry point.

Usage:
    presto data download [--all] [--source SOURCE] [--agree-iedb-terms]
    presto data list [--local] [--source SOURCE]
    presto data process [--dataset DATASET] [--filter FILTER]
    presto weights list [--registry PATH_OR_URL]
    presto weights download --name MODEL [--registry PATH_OR_URL]
    presto train synthetic [options]
    presto train unified [options]
    presto predict presentation [options]
    presto predict tile [options]
    presto predict recognition [options]
    presto predict chain [options]
    presto evaluate synthetic [options]
"""

import argparse
import sys
from pathlib import Path
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
    cmd_data_mhc_index_refresh,
    cmd_data_mhc_index_mouse_overlay,
    cmd_data_mhc_index_report,
    cmd_data_mhc_index_validate,
    cmd_data_mhc_index_resolve,
)
from .train import (
    cmd_train_synthetic,
    cmd_train_unified,
)
from .predict import (
    cmd_predict_presentation,
    cmd_predict_tile,
    cmd_predict_recognition,
    cmd_predict_chain,
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
    parser.add_argument(
        "--version", action="version", version=f"presto {__version__}"
    )

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
        "--outdir", "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded data (default: ./data)",
    )
    download_parser.add_argument(
        "--dataset", "-d",
        type=str,
        action="append",
        help="Specific dataset(s) to download. Can be repeated. "
             "Use 'presto data list' to see available datasets.",
    )
    download_parser.add_argument(
        "--source", "-s",
        type=str,
        action="append",
        choices=["iedb", "vdjdb", "mcpas", "imgt", "ipd_mhc", "10x", "pird", "stcrdab"],
        help="Download all datasets from specific source(s)",
    )
    download_parser.add_argument(
        "--category", "-c",
        type=str,
        action="append",
        choices=["binding", "tcell", "bcell", "tcr", "mhc_sequence", "elution", "vdj_genes"],
        help="Download all datasets of specific category",
    )
    download_parser.add_argument(
        "--all", "-a",
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
        "--force", "-f",
        action="store_true",
        help="Re-download even if files exist",
    )
    download_parser.add_argument(
        "--quiet", "-q",
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
        "--local", "-l",
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
        "--source", "-s",
        type=str,
        action="append",
        choices=["iedb", "vdjdb", "mcpas", "imgt", "ipd_mhc", "10x", "pird", "stcrdab"],
        help="Filter by source",
    )
    list_parser.add_argument(
        "--category", "-c",
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
        "--outdir", "-o",
        type=str,
        default="./data/processed",
        help="Output directory for processed data (default: ./data/processed)",
    )
    process_parser.add_argument(
        "--dataset", "-d",
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
        "--quiet", "-q",
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
        "--output", "-o",
        type=str,
        help="Output file (default: input_deduped.tsv)",
    )
    dedup_parser.add_argument(
        "--type", "-t",
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
        "--quiet", "-q",
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
        "--output", "-o",
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
        "--no-assay-csv",
        action="store_true",
        help="Disable per-assay CSV exports and only write the merged table",
    )
    merge_parser.add_argument(
        "--types", "-t",
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
        "--quiet", "-q",
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
        "--quiet", "-q",
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
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    mhc_build.set_defaults(func=cmd_data_mhc_index_build)

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
        "--output", "-o",
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
        "--output", "-o",
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
        "--output", "-o",
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
        "--output", "-o",
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
        "--force", "-f",
        action="store_true",
        help="Force re-download even if output file exists",
    )
    weights_download.set_defaults(func=cmd_weights_download)

    # ==========================================================================
    # train subcommand
    # ==========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Training workflows (synthetic demos and unified multi-source training)",
    )
    train_subparsers = train_parser.add_subparsers(
        dest="train_command",
        help="Training modes",
    )

    train_synth = train_subparsers.add_parser(
        "synthetic",
        help="Train on synthetic data (quick end-to-end demo)",
    )
    train_synth.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    train_synth.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_synth.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_synth.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_synth.add_argument("--d_model", type=int, default=128, help="Model dimension")
    train_synth.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    train_synth.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    train_synth.add_argument("--n_binding", type=int, default=200, help="Binding samples")
    train_synth.add_argument("--n_elution", type=int, default=100, help="Elution samples")
    train_synth.add_argument("--n_tcr", type=int, default=100, help="TCR samples")
    train_synth.add_argument("--data_dir", type=str, default=None, help="Data directory")
    train_synth.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    train_synth.add_argument("--run-dir", dest="run_dir", type=str, default=None, help="Run artifact directory")
    train_synth.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_synth.add_argument(
        "--use-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_true",
        default=True,
        help="Use learned uncertainty weighting over task losses",
    )
    train_synth.add_argument(
        "--no-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_false",
        help="Disable learned uncertainty weighting",
    )
    train_synth.add_argument(
        "--supervised-loss-aggregation",
        type=str,
        choices=["task_mean", "sample_weighted"],
        default="sample_weighted",
        help=(
            "How to combine supervised task losses: "
            "task_mean (equal per task) or sample_weighted "
            "(weight by in-batch labeled sample count per task)"
        ),
    )
    train_synth.add_argument("--use-pcgrad", action="store_true", help="Use PCGrad for multi-task gradients")
    train_synth.add_argument(
        "--consistency-cascade-weight",
        type=float,
        default=0.0,
        help="Weight for anti-saturation cascade prior (high presentation with low parent)",
    )
    train_synth.add_argument(
        "--consistency-assay-affinity-weight",
        type=float,
        default=0.0,
        help="Weight for KD/IC50/EC50 closeness regularization",
    )
    train_synth.add_argument(
        "--consistency-assay-presentation-weight",
        type=float,
        default=0.0,
        help="Weight for elution/MS vs presentation consistency",
    )
    train_synth.add_argument(
        "--consistency-no-b2m-weight",
        type=float,
        default=0.0,
        help="Weight for invalid chain-assembly prior (class I/II single-chain cases)",
    )
    train_synth.add_argument(
        "--consistency-tcell-context-weight",
        type=float,
        default=0.0,
        help="Weight for in-vitro >= ex-vivo T-cell context prior",
    )
    train_synth.add_argument(
        "--consistency-tcell-upstream-weight",
        type=float,
        default=0.0,
        help="Weight for T-cell outputs requiring strong upstream binding/presentation",
    )
    train_synth.add_argument(
        "--consistency-prob-margin",
        type=float,
        default=0.02,
        help="Shared margin used in probabilistic consistency constraints",
    )
    train_synth.add_argument(
        "--consistency-parent-low-threshold",
        type=float,
        default=0.1,
        help="Low-parent threshold used by anti-saturation presentation prior",
    )
    train_synth.add_argument(
        "--consistency-presentation-high-threshold",
        type=float,
        default=0.9,
        help="High-presentation threshold used by anti-saturation presentation prior",
    )
    train_synth.add_argument(
        "--consistency-affinity-fold-tolerance",
        type=float,
        default=2.0,
        help="Allowed KD/IC50/EC50 discrepancy fold before penalty (2.0 = within 2x)",
    )
    train_synth.add_argument(
        "--mhc-attention-sparsity-weight",
        type=float,
        default=0.0,
        help="Weight for binding latent MHC-attention support regularization",
    )
    train_synth.add_argument(
        "--mhc-attention-sparsity-min-residues",
        type=float,
        default=30.0,
        help="Lower target bound for effective attended MHC residues",
    )
    train_synth.add_argument(
        "--mhc-attention-sparsity-max-residues",
        type=float,
        default=60.0,
        help="Upper target bound for effective attended MHC residues",
    )
    train_synth.add_argument(
        "--tcell-in-vitro-margin",
        type=float,
        default=0.0,
        help="Required tcell-immunogenicity logit margin for in-vitro contexts",
    )
    train_synth.add_argument(
        "--tcell-ex-vivo-margin",
        type=float,
        default=0.0,
        help="Maximum tcell-immunogenicity logit margin for ex-vivo contexts",
    )
    train_synth.add_argument("--seed", type=int, default=42, help="Random seed")
    train_synth.set_defaults(func=cmd_train_synthetic)

    train_iedb = train_subparsers.add_parser(
        "unified",
        aliases=["iedb"],
        help="Train unified model on mixed-source data (IEDB/CEDAR + VDJdb + 10x)",
    )
    train_iedb.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    train_iedb.add_argument(
        "--profile",
        type=str,
        choices=["full", "canary", "diagnostic"],
        default="full",
        help=(
            "Training profile preset "
            "(canary: fast smoke run; diagnostic: richer coverage/flow/latent diagnostics)"
        ),
    )
    train_iedb.add_argument("--data-dir", dest="data_dir", type=str, default="./data", help="Data directory with downloaded datasets")
    train_iedb.add_argument(
        "--merged-tsv",
        type=str,
        default=None,
        help="Path to merged deduplicated TSV (default: <data-dir>/merged_deduped.tsv)",
    )
    train_iedb.add_argument(
        "--require-merged-input",
        dest="require_merged_input",
        action="store_true",
        default=True,
        help="Require merged TSV input (default: true)",
    )
    train_iedb.add_argument(
        "--allow-raw-fallback",
        dest="require_merged_input",
        action="store_false",
        help="Allow fallback to raw source exports when merged TSV is unavailable",
    )
    train_iedb.add_argument("--binding-file", type=str, default=None, help="Override path to IEDB MHC ligand export")
    train_iedb.add_argument("--tcell-file", type=str, default=None, help="Override path to IEDB T-cell export")
    train_iedb.add_argument("--cedar-binding-file", type=str, default=None, help="Optional path to CEDAR MHC ligand export")
    train_iedb.add_argument("--cedar-tcell-file", type=str, default=None, help="Optional path to CEDAR T-cell export")
    train_iedb.add_argument("--vdjdb-file", type=str, default=None, help="Override path to VDJdb export")
    train_iedb.add_argument(
        "--10x-file",
        dest="sc10x_file",
        type=str,
        default=None,
        help="Override path to 10x VDJ contig CSV/TSV",
    )
    train_iedb.add_argument("--index-csv", type=str, default=None, help="Optional built MHC index CSV")
    train_iedb.add_argument(
        "--strict-mhc-resolution",
        dest="strict_mhc_resolution",
        action="store_true",
        default=True,
        help="Require all non-ablation MHC alleles to resolve to amino-acid sequences (default: true)",
    )
    train_iedb.add_argument(
        "--allow-unresolved-mhc",
        dest="strict_mhc_resolution",
        action="store_false",
        help="Allow unresolved MHC alleles (debug only; unresolved MHC chains become empty sequences)",
    )
    train_iedb.add_argument(
        "--filter-unresolved-mhc",
        dest="filter_unresolved_mhc",
        action="store_true",
        default=False,
        help=(
            "Drop unresolved-MHC rows before dataset construction "
            "(resolved-only training subset)"
        ),
    )
    train_iedb.add_argument(
        "--no-filter-unresolved-mhc",
        dest="filter_unresolved_mhc",
        action="store_false",
        help="Disable unresolved-MHC row filtering",
    )
    train_iedb.add_argument(
        "--mhc-augmentation-samples",
        dest="mhc_augmentation_samples",
        type=int,
        default=60000,
        help="Number of MHC-only augmentation samples from index (0 to disable)",
    )
    train_iedb.add_argument("--max-binding", type=int, default=0, help="Max binding records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-kinetics", type=int, default=0, help="Max kinetics records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-stability", type=int, default=0, help="Max stability records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-processing", type=int, default=0, help="Max processing records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-elution", type=int, default=0, help="Max elution records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-tcell", type=int, default=0, help="Max T-cell records to load (<=0 means no limit)")
    train_iedb.add_argument("--max-vdjdb", type=int, default=0, help="Max VDJdb records to load (<=0 means no limit)")
    train_iedb.add_argument(
        "--cap-sampling",
        dest="cap_sampling",
        type=str,
        choices=["head", "reservoir"],
        default="reservoir",
        help=(
            "Sampling strategy when modality caps are set "
            "(head=first-N rows, reservoir=representative one-pass sample)"
        ),
    )
    train_iedb.add_argument(
        "--max-10x",
        dest="max_10x",
        type=int,
        default=0,
        help="Max 10x VDJ records to load (<=0 means no limit)",
    )
    train_iedb.add_argument(
        "--synthetic-pmhc-negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Primary synthetic non-binding pMHC ratio per real binding sample "
            "(also drives downstream elution/T-cell synthetic negatives)"
        ),
    )
    train_iedb.add_argument(
        "--synthetic-class-i-no-mhc-beta-negative-ratio",
        dest="synthetic_class_i_no_mhc_beta_negative_ratio",
        type=float,
        default=0.25,
        help="Additional class-I negatives without MHC beta chain (beta2m) per real class-I sample (0 disables)",
    )
    train_iedb.add_argument(
        "--synthetic-processing-negative-ratio",
        type=float,
        default=0.5,
        help="Synthetic processing negatives to add per real processing sample (0 disables)",
    )
    train_iedb.add_argument(
        "--synthetic-negative-min-nM",
        type=float,
        default=50000.0,
        help="Minimum synthetic weak-affinity value (nM)",
    )
    train_iedb.add_argument(
        "--synthetic-negative-max-nM",
        type=float,
        default=100000.0,
        help="Maximum synthetic weak-affinity value (nM)",
    )
    train_iedb.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    train_iedb.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_iedb.add_argument("--batch_size", type=int, default=512, help="Batch size")
    train_iedb.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (0 disables worker parallelism)",
    )
    train_iedb.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Enable pinned host memory in DataLoader (default: true)",
    )
    train_iedb.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory in DataLoader",
    )
    train_iedb.add_argument(
        "--balanced-batches",
        dest="balanced_batches",
        action="store_true",
        default=True,
        help="Balance train mini-batches by assay/source/label/allele strata (default: true)",
    )
    train_iedb.add_argument(
        "--no-balanced-batches",
        dest="balanced_batches",
        action="store_false",
        help="Disable balanced mini-batch sampling",
    )
    train_iedb.add_argument("--lr", type=float, default=2.8e-4, help="Learning rate")
    # Performance: AMP, compile, MIL cap
    train_iedb.add_argument(
        "--amp", dest="use_amp", action="store_true", default=True,
        help="Enable bf16 automatic mixed precision on CUDA (default: true)",
    )
    train_iedb.add_argument(
        "--no-amp", dest="use_amp", action="store_false",
        help="Disable bf16 automatic mixed precision",
    )
    train_iedb.add_argument(
        "--compile", dest="use_compile", action="store_true", default=False,
        help="Enable torch.compile for kernel fusion (default: false)",
    )
    train_iedb.add_argument(
        "--no-compile", dest="use_compile", action="store_false",
        help="Disable torch.compile",
    )
    train_iedb.add_argument(
        "--max-mil-instances", dest="max_mil_instances", type=int, default=128,
        help="Max MIL instances per batch (0=unlimited, default: 128)",
    )
    train_iedb.add_argument(
        "--max-batches", dest="max_batches", type=int, default=0,
        help="Max training batches per epoch (0=unlimited)",
    )
    train_iedb.add_argument(
        "--max-val-batches", dest="max_val_batches", type=int, default=0,
        help="Max validation batches per epoch (0=unlimited)",
    )
    train_iedb.add_argument("--d_model", type=int, default=128, help="Model dimension")
    train_iedb.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    train_iedb.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    train_iedb.add_argument("--checkpoint", type=str, default=None, help="Checkpoint output path")
    train_iedb.add_argument("--run-dir", dest="run_dir", type=str, default=None, help="Run artifact directory")
    train_iedb.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_iedb.add_argument(
        "--use-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_true",
        default=True,
        help="Use learned uncertainty weighting over task losses",
    )
    train_iedb.add_argument(
        "--no-uncertainty-weighting",
        dest="use_uncertainty_weighting",
        action="store_false",
        help="Disable learned uncertainty weighting",
    )
    train_iedb.add_argument(
        "--supervised-loss-aggregation",
        type=str,
        choices=["task_mean", "sample_weighted"],
        default="sample_weighted",
        help=(
            "How to combine supervised task losses: "
            "task_mean (equal per task) or sample_weighted "
            "(weight by in-batch labeled sample count per task)"
        ),
    )
    train_iedb.add_argument(
        "--profile-performance",
        dest="profile_performance",
        action="store_true",
        default=True,
        help="Record epoch performance breakdown (data wait/compute/backward/optimizer)",
    )
    train_iedb.add_argument(
        "--no-profile-performance",
        dest="profile_performance",
        action="store_false",
        help="Disable epoch performance breakdown instrumentation",
    )
    train_iedb.add_argument(
        "--perf-log-interval-batches",
        type=int,
        default=100,
        help=(
            "Emit rolling in-epoch perf breakdown every N train batches "
            "(0 disables rolling perf logs)"
        ),
    )
    train_iedb.add_argument("--use-pcgrad", action="store_true", help="Use PCGrad for multi-task gradients")
    train_iedb.add_argument(
        "--consistency-cascade-weight",
        type=float,
        default=0.2,
        help="Weight for anti-saturation cascade prior (high presentation with low parent)",
    )
    train_iedb.add_argument(
        "--consistency-assay-affinity-weight",
        type=float,
        default=0.1,
        help="Weight for KD/IC50/EC50 closeness regularization",
    )
    train_iedb.add_argument(
        "--consistency-assay-presentation-weight",
        type=float,
        default=0.1,
        help="Weight for elution/MS vs presentation consistency",
    )
    train_iedb.add_argument(
        "--consistency-no-b2m-weight",
        type=float,
        default=0.5,
        help="Weight for invalid chain-assembly prior (class I/II single-chain cases)",
    )
    train_iedb.add_argument(
        "--consistency-tcell-context-weight",
        type=float,
        default=0.05,
        help="Weight for in-vitro >= ex-vivo T-cell context prior",
    )
    train_iedb.add_argument(
        "--consistency-tcell-upstream-weight",
        type=float,
        default=0.2,
        help="Weight for T-cell outputs requiring strong upstream binding/presentation",
    )
    train_iedb.add_argument(
        "--consistency-prob-margin",
        type=float,
        default=0.02,
        help="Shared margin used in probabilistic consistency constraints",
    )
    train_iedb.add_argument(
        "--consistency-parent-low-threshold",
        type=float,
        default=0.1,
        help="Low-parent threshold used by anti-saturation presentation prior",
    )
    train_iedb.add_argument(
        "--consistency-presentation-high-threshold",
        type=float,
        default=0.9,
        help="High-presentation threshold used by anti-saturation presentation prior",
    )
    train_iedb.add_argument(
        "--consistency-affinity-fold-tolerance",
        type=float,
        default=2.0,
        help="Allowed KD/IC50/EC50 discrepancy fold before penalty (2.0 = within 2x)",
    )
    train_iedb.add_argument(
        "--mhc-attention-sparsity-weight",
        type=float,
        default=0.0,
        help="Weight for binding latent MHC-attention support regularization",
    )
    train_iedb.add_argument(
        "--mhc-attention-sparsity-min-residues",
        type=float,
        default=30.0,
        help="Lower target bound for effective attended MHC residues",
    )
    train_iedb.add_argument(
        "--mhc-attention-sparsity-max-residues",
        type=float,
        default=60.0,
        help="Upper target bound for effective attended MHC residues",
    )
    train_iedb.add_argument(
        "--tcell-in-vitro-margin",
        type=float,
        default=0.1,
        help="Required tcell-immunogenicity logit margin for in-vitro contexts",
    )
    train_iedb.add_argument(
        "--tcell-ex-vivo-margin",
        type=float,
        default=0.0,
        help="Maximum tcell-immunogenicity logit margin for ex-vivo contexts",
    )
    train_iedb.add_argument(
        "--track-probe-affinity",
        dest="track_probe_affinity",
        action="store_true",
        default=True,
        help=(
            "Track fixed probe pMHC affinities every epoch and emit "
            "probe_affinity_over_epochs.{csv,json,png}"
        ),
    )
    train_iedb.add_argument(
        "--no-track-probe-affinity",
        dest="track_probe_affinity",
        action="store_false",
        help="Disable fixed probe affinity tracking and plotting",
    )
    train_iedb.add_argument(
        "--probe-peptide",
        type=str,
        default="SLLQHLIGL",
        help="Peptide sequence used for fixed per-epoch probe tracking",
    )
    train_iedb.add_argument(
        "--probe-alleles",
        type=str,
        default="HLA-A*02:01,HLA-A*24:02",
        help="Comma-separated alleles for probe tracking",
    )
    train_iedb.add_argument(
        "--probe-plot-file",
        type=str,
        default="probe_affinity_over_epochs.png",
        help="Filename (inside run-dir) for probe affinity plot",
    )
    train_iedb.add_argument(
        "--track-probe-motif-scan",
        dest="track_probe_motif_scan",
        action="store_true",
        default=True,
        help=(
            "Track probe peptide single-residue substitution scans at selected positions "
            "and log motif-oriented metrics per epoch"
        ),
    )
    train_iedb.add_argument(
        "--no-track-probe-motif-scan",
        dest="track_probe_motif_scan",
        action="store_false",
        help="Disable probe motif substitution-scan diagnostics",
    )
    train_iedb.add_argument(
        "--motif-scan-positions",
        type=str,
        default="2,9",
        help="Comma-separated 1-based peptide positions for probe motif scanning",
    )
    train_iedb.add_argument(
        "--motif-scan-amino-acids",
        type=str,
        default="ACDEFGHIKLMNPQRSTVWY",
        help="Amino-acid alphabet used for probe motif substitutions",
    )
    train_iedb.add_argument(
        "--track-pmhc-flow",
        dest="track_pmhc_flow",
        action="store_true",
        default=True,
        help=(
            "Track peptide-MHC information flow with counterfactual shuffles "
            "(real vs MHC-shuffled vs peptide-shuffled vs both)"
        ),
    )
    train_iedb.add_argument(
        "--no-track-pmhc-flow",
        dest="track_pmhc_flow",
        action="store_false",
        help="Disable peptide-MHC information-flow diagnostics",
    )
    train_iedb.add_argument(
        "--pmhc-flow-batches",
        type=int,
        default=2,
        help="Validation batches per epoch to use for pMHC information-flow diagnostics",
    )
    train_iedb.add_argument(
        "--pmhc-flow-max-samples",
        type=int,
        default=512,
        help="Max validation samples per epoch to evaluate for pMHC information-flow diagnostics",
    )
    train_iedb.add_argument(
        "--track-output-latent-stats",
        dest="track_output_latent_stats",
        action="store_true",
        default=True,
        help=(
            "Track validation statistics for output heads and latent vectors "
            "(means/variances/norms)"
        ),
    )
    train_iedb.add_argument(
        "--no-track-output-latent-stats",
        dest="track_output_latent_stats",
        action="store_false",
        help="Disable output/latent diagnostic tracking",
    )
    train_iedb.add_argument(
        "--output-latent-stats-batches",
        type=int,
        default=2,
        help="Validation batches per epoch for output/latent diagnostics",
    )
    train_iedb.add_argument(
        "--output-latent-stats-max-samples",
        type=int,
        default=512,
        help="Max validation samples per epoch for output/latent diagnostics",
    )
    train_iedb.add_argument("--uniprot-negative-ratio", dest="uniprot_negative_ratio",
                            type=float, default=0.1,
                            help="Ratio of UniProt negative samples to add (default: 0.1)")
    train_iedb.add_argument("--max-uniprot", dest="max_uniprot",
                            type=int, default=0,
                            help="Max UniProt negative samples (0 = unlimited)")
    train_iedb.add_argument("--seed", type=int, default=42, help="Random seed")
    train_iedb.add_argument("--device", type=str, default=None, help="Device")
    train_iedb.set_defaults(func=cmd_train_unified)

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
    predict_presentation.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_presentation.add_argument("--peptide", type=str, required=True, help="Peptide sequence")
    predict_presentation.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_presentation.add_argument("--mhc-sequence", type=str, default=None, help="MHC alpha sequence")
    predict_presentation.add_argument("--mhc-b-sequence", type=str, default=None, help="MHC beta sequence")
    predict_presentation.add_argument("--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class")
    predict_presentation.add_argument("--species", type=str, default=None, help="Species label for class-I beta2m resolution")
    predict_presentation.add_argument("--mhc-species", type=str, default=None, help="Override MHC species latent/probability path")
    predict_presentation.add_argument("--immune-species", type=str, default=None, help="Override immune-system species conditioning")
    predict_presentation.add_argument("--species-of-origin", type=str, default=None, help="Override peptide species-of-origin latent")
    predict_presentation.add_argument("--flank-n", type=str, default=None, help="N-terminal flank")
    predict_presentation.add_argument("--flank-c", type=str, default=None, help="C-terminal flank")
    predict_presentation.add_argument("--index-csv", type=str, default=None, help="Optional built MHC index CSV for allele sequence lookup")
    predict_presentation.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_presentation.add_argument("--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory")
    predict_presentation.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    predict_presentation.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    predict_presentation.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
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
    predict_tile.add_argument("--protein-file", type=str, default=None, help="Path to FASTA/plain sequence file")
    predict_tile.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_tile.add_argument("--mhc-sequence", type=str, default=None, help="MHC alpha sequence")
    predict_tile.add_argument("--mhc-b-sequence", type=str, default=None, help="MHC beta sequence")
    predict_tile.add_argument("--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class")
    predict_tile.add_argument("--species", type=str, default=None, help="Species label for class-I beta2m resolution")
    predict_tile.add_argument("--mhc-species", type=str, default=None, help="Override MHC species latent/probability path")
    predict_tile.add_argument("--immune-species", type=str, default=None, help="Override immune-system species conditioning")
    predict_tile.add_argument("--species-of-origin", type=str, default=None, help="Override peptide species-of-origin latent")
    predict_tile.add_argument("--min-length", type=int, default=8, help="Minimum peptide length for tiling")
    predict_tile.add_argument("--max-length", type=int, default=15, help="Maximum peptide length for tiling")
    predict_tile.add_argument("--flank-size", type=int, default=15, help="Context flank size on each side")
    predict_tile.add_argument("--batch-size", type=int, default=128, help="Batch size for tiled inference")
    predict_tile.add_argument("--top-k", type=int, default=100, help="Keep top-k hits (<=0 keeps all)")
    predict_tile.add_argument(
        "--sort-by",
        type=str,
        choices=["presentation", "binding", "processing"],
        default="presentation",
        help="Sort metric for reported tiled hits",
    )
    predict_tile.add_argument("--index-csv", type=str, default=None, help="Optional built MHC index CSV for allele sequence lookup")
    predict_tile.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_tile.add_argument("--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory")
    predict_tile.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    predict_tile.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    predict_tile.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
    predict_tile.add_argument("--device", type=str, default=None, help="Device")
    predict_tile.add_argument("--json", action="store_true", help="Output JSON")
    predict_tile.add_argument("--output", type=str, default=None, help="Output path")
    predict_tile.set_defaults(func=cmd_predict_tile)

    predict_recognition = predict_subparsers.add_parser(
        "recognition",
        help="Predict TCR-pMHC recognition/immunogenicity",
    )
    predict_recognition.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_recognition.add_argument("--peptide", type=str, required=True, help="Peptide sequence")
    predict_recognition.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_recognition.add_argument("--mhc-sequence", type=str, default=None, help="MHC alpha sequence")
    predict_recognition.add_argument("--mhc-b-sequence", type=str, default=None, help="MHC beta sequence")
    predict_recognition.add_argument("--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class")
    predict_recognition.add_argument("--species", type=str, default=None, help="Species label for class-I beta2m resolution")
    predict_recognition.add_argument("--mhc-species", type=str, default=None, help="Override MHC species latent/probability path")
    predict_recognition.add_argument("--immune-species", type=str, default=None, help="Override immune-system species conditioning")
    predict_recognition.add_argument("--species-of-origin", type=str, default=None, help="Override peptide species-of-origin latent")
    predict_recognition.add_argument("--tcr-alpha", type=str, required=False, help="TCR alpha sequence")
    predict_recognition.add_argument("--tcr-beta", type=str, required=False, help="TCR beta sequence")
    predict_recognition.add_argument("--index-csv", type=str, default=None, help="Optional built MHC index CSV for allele sequence lookup")
    predict_recognition.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_recognition.add_argument("--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory")
    predict_recognition.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    predict_recognition.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    predict_recognition.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
    predict_recognition.add_argument("--device", type=str, default=None, help="Device")
    predict_recognition.add_argument("--json", action="store_true", help="Output JSON")
    predict_recognition.add_argument("--output", type=str, default=None, help="Output path")
    predict_recognition.set_defaults(func=cmd_predict_recognition)

    predict_chain = predict_subparsers.add_parser(
        "chain",
        help="Classify chain attributes from sequence",
    )
    predict_chain.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_chain.add_argument("--sequence", type=str, required=True, help="Chain sequence")
    predict_chain.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    predict_chain.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    predict_chain.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
    predict_chain.add_argument("--device", type=str, default=None, help="Device")
    predict_chain.add_argument("--json", action="store_true", help="Output JSON")
    predict_chain.add_argument("--output", type=str, default=None, help="Output path")
    predict_chain.set_defaults(func=cmd_predict_chain)

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
    eval_synth.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    eval_synth.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    eval_synth.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
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
    if args.command == "train" and getattr(args, "train_command", None) is None:
        parser.parse_args(["train", "--help"])
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
