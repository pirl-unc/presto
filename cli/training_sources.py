"""One training-source argument contract for both canonical entry points."""


def add_training_source_arguments(parser):
    parser.add_argument(
        "--model-kind",
        choices=("legacy", "modular"),
        default="legacy",
        help="Select the existing model or the versioned small modular model",
    )
    parser.add_argument(
        "--observations-manifest",
        help="Hash-pinned normalized source/model/training manifest for modular mode",
    )
    parser.add_argument(
        "--modular-preflight",
        action="store_true",
        help="Validate modular source and label support without fitting",
    )
    parser.add_argument(
        "--modular-resume",
        action="store_true",
        help=(
            "Resume the same frozen modular experiment at its last durable update; "
            "requires --run-dir"
        ),
    )
    parser.add_argument(
        "--data-source",
        dest="data_source",
        type=str,
        choices=["merged_tsv", "hitlist", "scoped_bulk"],
        default="merged_tsv",
        help=(
            "Select exactly one training source: 'merged_tsv' loads the canonical merged "
            "table; 'hitlist' requires a verified --hitlist-bundle and covers "
            "binding/stability/kinetics/elution only; scoped_bulk loads verified search-scoped "
            "detectability proxies."
        ),
    )
    parser.add_argument(
        "--hitlist-bundle",
        help="Frozen Hitlist bundle directory; verified and shared through datacache",
    )
    parser.add_argument(
        "--hitlist-bundle-sha256",
        help="Expected SHA-256 of the frozen bundle's manifest.json",
    )
    parser.add_argument(
        "--frozen-assignment-file",
        help=(
            "Optional frozen membership parquet (observation IDs or scoped sample IDs); "
            "requires SHA-256"
        ),
    )
    parser.add_argument("--frozen-assignment-sha256", help="SHA-256 of the assignment parquet")
    parser.add_argument(
        "--frozen-partition-policy",
        default="peptide_disjoint",
        choices=[
            "peptide_disjoint",
            "pmhc_disjoint",
            "independent_experiments",
            "specimen_disjoint",
            "donor_disjoint",
            "whole_study",
            "report_only",
        ],
        help="Required measured-root independence policy; unknown required identities fail",
    )
    parser.add_argument(
        "--hitlist-component-allocation",
        default="relative_deficit_v2",
        choices=["absolute_deficit_v1", "relative_deficit_v2"],
        help=(
            "Frozen Hitlist whole-component allocation: v2 balances fractional remaining "
            "capacity; v1 reproduces the historical absolute-deficit policy. "
            "Scoped-search membership retains its own frozen v1 policy."
        ),
    )
    parser.add_argument(
        "--scoped-bundle", help="Optional local PXD bundle; requires its manifest hash"
    )
    parser.add_argument("--scoped-bundle-sha256", help="Expected local scoped manifest SHA-256")
    parser.add_argument(
        "--max-scoped-bulk",
        type=int,
        default=0,
        help="Cap globally collapsed scoped peptides (0 means all)",
    )
