"""Prediction CLI commands."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..data import AlleleResolver
from ..inference.predictor import Predictor


def _load_allele_sequences(
    imgt_fasta: Optional[str],
    ipd_mhc_dir: Optional[str],
) -> Dict[str, str]:
    if not imgt_fasta and not ipd_mhc_dir:
        return {}

    resolver = AlleleResolver(imgt_fasta=imgt_fasta, ipd_mhc_dir=ipd_mhc_dir)
    sequences: Dict[str, str] = {name: record.sequence for name, record in resolver.records.items()}
    for alias, canonical in resolver._aliases.items():
        sequences[alias] = resolver.records[canonical].sequence
    return sequences


def _build_predictor(args: Any) -> Predictor:
    if not args.checkpoint:
        raise ValueError("Missing --checkpoint")

    allele_sequences = _load_allele_sequences(args.imgt_fasta, args.ipd_mhc_dir)
    return Predictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        allele_sequences=allele_sequences,
        device=args.device,
    )


def _emit_output(payload: dict, args: Any) -> None:
    if args.json or args.output:
        output = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
        return

    # Human-readable output
    for key, value in payload.items():
        print(f"{key}: {value}")


def cmd_predict_presentation(args: Any) -> int:
    """Predict presentation probabilities."""
    predictor = _build_predictor(args)
    result = predictor.predict_presentation(
        peptide=args.peptide,
        allele=args.allele,
        mhc_sequence=args.mhc_sequence,
        mhc_b_sequence=args.mhc_b_sequence,
        mhc_class=args.mhc_class,
        flank_n=args.flank_n,
        flank_c=args.flank_c,
    )
    _emit_output(asdict(result), args)
    return 0


def cmd_predict_recognition(args: Any) -> int:
    """Predict TCR-pMHC recognition and immunogenicity."""
    predictor = _build_predictor(args)
    result = predictor.predict_recognition(
        peptide=args.peptide,
        allele=args.allele,
        mhc_sequence=args.mhc_sequence,
        tcr_alpha=args.tcr_alpha,
        tcr_beta=args.tcr_beta,
        mhc_class=args.mhc_class,
    )
    _emit_output(asdict(result), args)
    return 0


def cmd_predict_chain(args: Any) -> int:
    """Predict chain attributes from sequence."""
    predictor = _build_predictor(args)
    result = predictor.classify_chain(args.sequence)
    _emit_output(asdict(result), args)
    return 0
