"""PRESTO CLI main entry point.

Usage:
    presto data download [--all] [--source SOURCE] [--agree-iedb-terms]
    presto data list [--local] [--source SOURCE]
    presto data process [--dataset DATASET] [--filter FILTER]
    presto train synthetic [options]
    presto train curriculum [options]
    presto predict presentation [options]
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
)
from .train import cmd_train_synthetic, cmd_train_curriculum
from .predict import (
    cmd_predict_presentation,
    cmd_predict_recognition,
    cmd_predict_chain,
)
from .evaluate import cmd_evaluate_synthetic


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="presto",
        description="PRESTO - Peptide-Receptor Embedding for Shared T-cell Ontology",
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

    # ==========================================================================
    # train subcommand
    # ==========================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Training workflows (synthetic or curriculum demos)",
    )
    train_subparsers = train_parser.add_subparsers(
        dest="train_command",
        help="Training modes",
    )

    train_synth = train_subparsers.add_parser(
        "synthetic",
        help="Train on synthetic data (quick end-to-end demo)",
    )
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
    train_synth.add_argument("--seed", type=int, default=42, help="Random seed")
    train_synth.set_defaults(func=cmd_train_synthetic)

    train_curr = train_subparsers.add_parser(
        "curriculum",
        help="Run curriculum training demo (synthetic)",
    )
    train_curr.add_argument("--epochs", type=int, default=40, help="Total epochs")
    train_curr.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_curr.add_argument("--d_model", type=int, default=128, help="Model dimension")
    train_curr.add_argument("--n_samples", type=int, default=1000, help="Samples per task")
    train_curr.add_argument("--seed", type=int, default=42, help="Random seed")
    train_curr.add_argument("--device", type=str, default=None, help="Device")
    train_curr.set_defaults(func=cmd_train_curriculum)

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
    predict_presentation.add_argument("--flank-n", type=str, default=None, help="N-terminal flank")
    predict_presentation.add_argument("--flank-c", type=str, default=None, help="C-terminal flank")
    predict_presentation.add_argument("--imgt-fasta", type=str, default=None, help="IMGT/HLA FASTA")
    predict_presentation.add_argument("--ipd-mhc-dir", type=str, default=None, help="IPD-MHC directory")
    predict_presentation.add_argument("--d-model", dest="d_model", type=int, default=None, help="Model dimension")
    predict_presentation.add_argument("--n-layers", dest="n_layers", type=int, default=None, help="Transformer layers")
    predict_presentation.add_argument("--n-heads", dest="n_heads", type=int, default=None, help="Attention heads")
    predict_presentation.add_argument("--device", type=str, default=None, help="Device")
    predict_presentation.add_argument("--json", action="store_true", help="Output JSON")
    predict_presentation.add_argument("--output", type=str, default=None, help="Output path")
    predict_presentation.set_defaults(func=cmd_predict_presentation)

    predict_recognition = predict_subparsers.add_parser(
        "recognition",
        help="Predict TCR-pMHC recognition/immunogenicity",
    )
    predict_recognition.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    predict_recognition.add_argument("--peptide", type=str, required=True, help="Peptide sequence")
    predict_recognition.add_argument("--allele", type=str, default=None, help="MHC allele name")
    predict_recognition.add_argument("--mhc-sequence", type=str, default=None, help="MHC alpha sequence")
    predict_recognition.add_argument("--mhc-class", type=str, choices=["I", "II"], default=None, help="MHC class")
    predict_recognition.add_argument("--tcr-alpha", type=str, required=False, help="TCR alpha sequence")
    predict_recognition.add_argument("--tcr-beta", type=str, required=False, help="TCR beta sequence")
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
