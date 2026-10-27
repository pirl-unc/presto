"""The one public training argument contract; all scientific settings are pinned."""

import argparse


def add_training_source_arguments(parser):
    parser.add_argument(
        "--observations-manifest",
        "--manifest",
        dest="observations_manifest",
        required=True,
        help="Hash-pinned observations, model and training configuration",
    )
    parser.add_argument(
        "--preflight",
        "--modular-preflight",
        dest="modular_preflight",
        action="store_true",
        help="Validate the source, target and model contract without fitting",
    )
    parser.add_argument(
        "--resume",
        "--modular-resume",
        dest="modular_resume",
        action="store_true",
        help="Resume the same frozen run; requires --run-dir",
    )
    parser.add_argument("--run-dir", help="Registered timestamped experiment directory")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--model-kind",
        choices=("shared", "modular"),
        default="shared",
        help=argparse.SUPPRESS,
    )
