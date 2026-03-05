"""Prediction CLI commands."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..data import AlleleResolver
from ..data.mhc_index import load_mhc_index
from ..inference.predictor import Predictor


def _load_allele_sequences(
    imgt_fasta: Optional[str],
    ipd_mhc_dir: Optional[str],
    index_csv: Optional[str] = None,
) -> Dict[str, str]:
    sequences: Dict[str, str] = {}

    if imgt_fasta or ipd_mhc_dir:
        resolver = AlleleResolver(imgt_fasta=imgt_fasta, ipd_mhc_dir=ipd_mhc_dir)
        sequences.update({name: record.sequence for name, record in resolver.records.items()})
        for alias, canonical in resolver._aliases.items():
            sequences[alias] = resolver.records[canonical].sequence

    if index_csv:
        records = load_mhc_index(index_csv)
        for record in records.values():
            for token in {record.normalized, record.allele_raw}:
                if not token:
                    continue
                sequences[token] = record.sequence
                if ":" in token:
                    parts = token.split(":")
                    for i in range(1, len(parts)):
                        sequences.setdefault(":".join(parts[:i]), record.sequence)

    if not sequences:
        return {}
    return sequences


def _build_predictor(args: Any) -> Predictor:
    if not args.checkpoint:
        raise ValueError("Missing --checkpoint")

    allele_sequences = _load_allele_sequences(
        getattr(args, "imgt_fasta", None),
        getattr(args, "ipd_mhc_dir", None),
        getattr(args, "index_csv", None),
    )
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
        species=args.species,
        mhc_species=getattr(args, "mhc_species", None),
        immune_species=getattr(args, "immune_species", None),
        species_of_origin=getattr(args, "species_of_origin", None),
        flank_n=args.flank_n,
        flank_c=args.flank_c,
    )
    _emit_output(asdict(result), args)
    return 0


def cmd_predict_recognition(args: Any) -> int:
    """Predict TCR-pMHC recognition and immunogenicity."""
    predictor = _build_predictor(args)
    raise NotImplementedError(
        "predict recognition is reserved for a future TCR pathway and is "
        "currently disabled in canonical Presto."
    )


def _read_single_sequence(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Protein file is empty: {path}")

    if text.startswith(">"):
        seq_parts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line)
        sequence = "".join(seq_parts)
    else:
        sequence = "".join(line.strip() for line in text.splitlines())

    sequence = sequence.strip()
    if not sequence:
        raise ValueError(f"No sequence content found in: {path}")
    return sequence


def cmd_predict_tile(args: Any) -> int:
    """Tile presentation predictions across all protein subsequences."""
    if bool(args.protein_sequence) == bool(args.protein_file):
        raise ValueError("Provide exactly one of --protein-sequence or --protein-file")
    if args.protein_file:
        protein_sequence = _read_single_sequence(Path(args.protein_file))
    else:
        protein_sequence = args.protein_sequence

    predictor = _build_predictor(args)
    result = predictor.predict_tiled_presentation(
        protein_sequence=protein_sequence,
        allele=args.allele,
        mhc_sequence=args.mhc_sequence,
        mhc_b_sequence=args.mhc_b_sequence,
        mhc_class=args.mhc_class,
        species=args.species,
        mhc_species=getattr(args, "mhc_species", None),
        immune_species=getattr(args, "immune_species", None),
        species_of_origin=getattr(args, "species_of_origin", None),
        min_length=args.min_length,
        max_length=args.max_length,
        flank_size=args.flank_size,
        batch_size=args.batch_size,
        top_k=args.top_k,
        sort_by=args.sort_by,
    )
    _emit_output(asdict(result), args)
    return 0


def cmd_predict_chain(args: Any) -> int:
    """Predict chain attributes from sequence."""
    predictor = _build_predictor(args)
    result = predictor.classify_chain(args.sequence)
    _emit_output(asdict(result), args)
    return 0
