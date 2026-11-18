"""Installed commands for the shared observation model and pinned source workflow."""

from dataclasses import replace
import gzip
from itertools import islice
import json
from pathlib import Path
import shutil
import tempfile
import textwrap

from ..data.modular import Observation
from ..data.modular_context import project_apc_context
from ..data.modular_sources import ObservationIndex, file_sha256


def json_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _checkpoint(args):
    from ..inference.modular import ModularPredictor

    digest = file_sha256(args.checkpoint)
    if args.checkpoint_sha256 and digest != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 disagrees with the requested checkpoint")
    return ModularPredictor(args.checkpoint, device=args.device), digest


def _batched_predict_rows(path, batch_size):
    iterator = json_rows(path)
    while True:
        chunk = [Observation.from_dict(raw) for raw in islice(iterator, batch_size)]
        if not chunk:
            break
        if any(
            row.split != "predict" or row.value is not None or row.identity_labels for row in chunk
        ):
            raise ValueError("prediction inputs must be nonempty, label-free predict observations")
        yield chunk


def cmd_predict_observations(args):
    batch_size = getattr(args, "batch_size", 64)
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    predictor, digest = _checkpoint(args)
    inputs_digest = file_sha256(args.inputs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    wrote_any = False
    try:
        with output.open("x", encoding="utf-8") as stream:
            created = True
            stream.write("{\n")
            stream.write(f'  "checkpoint_sha256": {json.dumps(digest, allow_nan=False)},\n')
            stream.write(f'  "inputs_sha256": {json.dumps(inputs_digest, allow_nan=False)},\n')
            stream.write(
                f'  "unsupported_apc_policy": '
                f'{json.dumps(args.unsupported_apc_policy, allow_nan=False)},\n'
            )
            stream.write('  "predictions": [')
            for batch in _batched_predict_rows(args.inputs, batch_size):
                predictions = predictor.predict(
                    batch, unsupported_apc_policy=args.unsupported_apc_policy
                )
                for item in predictions:
                    stream.write(",\n" if wrote_any else "\n")
                    stream.write(
                        textwrap.indent(json.dumps(item, indent=2, allow_nan=False), "    ")
                    )
                    wrote_any = True
                stream.flush()
            if not wrote_any:
                raise ValueError(
                    "prediction inputs must be nonempty, label-free predict observations"
                )
            stream.write("\n  ]\n}\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    return 0


def _evaluation_manifest(path, split):
    """Select declared source partitions before opening any observation payload."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    sources = manifest.get("sources", [])
    if not sources or any(
        source.get("split") not in {"train", "validation", "test"} for source in sources
    ):
        raise ValueError("evaluation requires an explicit split on every manifest source")
    selected = [
        {**source, "path": str((path.parent / source["path"]).resolve())}
        for source in sources
        if source["split"] == split
    ]
    if not selected:
        raise ValueError(f"no sources declared for {split}")
    return {**manifest, "sources": selected}


def cmd_evaluate_observations(args):
    from ..evaluation.presentation import paired_metrics
    from ..training.modular_evaluation import evaluate, summarize_strata

    if args.batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    manifest = _evaluation_manifest(args.manifest, args.split)
    predictor, digest = _checkpoint(args)
    with tempfile.TemporaryDirectory(prefix="presto-evaluate-") as temporary:
        selected = Path(temporary) / "manifest.json"
        write_json(selected, manifest)
        index = ObservationIndex(selected)
        if {key.split(":")[0] for key in index.partitions} != {args.split}:
            raise ValueError("source contents disagree with the declared evaluation split")
        selected_indices = sorted({i for values in index.partitions.values() for i in values})
        rows = [index[i] for i in selected_indices]
        projected = 0
        if args.unsupported_apc_policy == "unknown":
            config = predictor.model.config
            updated = []
            for row in rows:
                context = project_apc_context(
                    row.context,
                    cell_types=config.cell_types,
                    cytokines=config.cytokines,
                    processing_pathways=config.processing_pathways,
                    interventions=config.interventions,
                )
                projected += context is not row.context
                updated.append(replace(row, context=context))
            rows = updated
        metrics, records = evaluate(
            predictor.model,
            rows,
            list(range(len(rows))),
            predictor.scales,
            batch_size=args.batch_size,
            require_support=True,
        )
    strata = summarize_strata(records)
    pairs = paired_metrics(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        predictions = output / "predictions.jsonl.gz"
        with gzip.open(predictions, "wt") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        write_json(output / "metrics.json", metrics)
        write_json(output / "strata.json", strata)
        write_json(output / "pairs.json", pairs)
        write_json(
            output / "receipt.json",
            {
                "checkpoint_sha256": digest,
                "manifest_sha256": file_sha256(args.manifest),
                "selected_sources": manifest["sources"],
                "split": args.split,
                "rows": len(rows),
                "unsupported_apc_policy": args.unsupported_apc_policy,
                "projected_rows": projected,
                "predictions_sha256": file_sha256(predictions),
                "test_accessed": args.split == "test",
            },
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


def cmd_verify_corpus(args):
    from ..data.hitlist_corpus import verify_corpus

    print(json.dumps(verify_corpus(args.corpus_dir, pin=args.pin), indent=2))
    return 0


def cmd_export_presentation(args):
    from ..data.hitlist_corpus import export_presentation

    export_presentation(args.corpus_dir, args.output_dir, pin=args.pin)
    return 0


def cmd_prepare_presentation(args):
    from ..data.presentation_preparation import prepare_from_manifest

    prepare_from_manifest(args.manifest, args.output_dir, admission_only=args.admission_only)
    return 0


def cmd_freeze_identities(args):
    from ..data.presentation_identities import freeze_identities

    freeze_identities(args.manifest, args.output_dir)
    return 0


def cmd_audit_presentation(args):
    from ..data.presentation_preparation import audit_prepared

    receipt = audit_prepared(args.prepared_dir)
    with Path(args.output).open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return 0


def add_observation_commands(data, predict, evaluate):
    """Keep the shared workflow in one place; old CLI commands remain compatible."""
    for name, handler in (
        ("verify-corpus", cmd_verify_corpus),
        ("export-presentation", cmd_export_presentation),
    ):
        parser = data.add_parser(name, help=handler.__name__.replace("cmd_", ""))
        parser.add_argument("--corpus-dir", required=True)
        parser.add_argument("--pin", help="Pinned corpus contract JSON (default: corrected v4)")
        if name == "export-presentation":
            parser.add_argument("--output-dir", required=True)
        parser.set_defaults(func=handler)
    parser = data.add_parser("prepare-presentation", help="Prepare protected both-class pairs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admission-only", action="store_true")
    parser.set_defaults(func=cmd_prepare_presentation)
    parser = data.add_parser("freeze-identities", help="Pin TRAIN/VALIDATION peptide identities")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(func=cmd_freeze_identities)
    parser = data.add_parser(
        "audit-presentation", help="Audit written pairs and identity protection"
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.set_defaults(func=cmd_audit_presentation)
    for parent, handler in (
        (predict, cmd_predict_observations),
        (evaluate, cmd_evaluate_observations),
    ):
        parser = parent.add_parser("observations", help="Use the shared observation model")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--checkpoint-sha256")
        parser.add_argument("--device", default="cpu")
        parser.add_argument(
            "--unsupported-apc-policy", choices=("error", "unknown"), default="error"
        )
        parser.add_argument("--batch-size", type=int, default=64)
        if parent is predict:
            parser.add_argument("--inputs", required=True, help="Label-free JSONL or JSONL.gz")
            parser.add_argument("--output", required=True)
        else:
            parser.add_argument("--manifest", required=True)
            parser.add_argument("--split", choices=("validation", "test"), default="validation")
            parser.add_argument("--output-dir", required=True)
        parser.set_defaults(func=handler)
