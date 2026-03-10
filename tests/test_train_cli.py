"""Tests for train CLI parser wiring."""

from presto.cli import train as train_cli
from presto.cli.main import create_parser


def test_parser_wires_train_iedb_run_dir():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--run-dir",
            "/tmp/unified",
        ]
    )
    assert args.func is train_cli.cmd_train_unified
    assert args.run_dir == "/tmp/unified"


def test_parser_train_iedb_default_record_caps_unlimited():
    parser = create_parser()
    args = parser.parse_args(["train", "unified"])
    assert args.func is train_cli.cmd_train_unified
    assert args.profile == "full"
    assert args.require_merged_input is True
    assert args.balanced_batches is True
    assert args.merged_tsv is None
    assert args.cedar_binding_file is None
    assert args.cedar_tcell_file is None
    assert args.max_binding == 0
    assert args.max_kinetics == 0
    assert args.max_stability == 0
    assert args.max_processing == 0
    assert args.max_elution == 0
    assert args.max_tcell == 0
    assert args.num_workers == 4
    assert args.pin_memory is True
    assert args.profile_performance is True
    assert args.supervised_loss_aggregation == "task_mean"
    assert args.perf_log_interval_batches == 100
    assert args.track_pmhc_flow is True
    assert args.pmhc_flow_batches == 2
    assert args.pmhc_flow_max_samples == 512
    assert args.filter_unresolved_mhc is True
    assert args.strict_mhc_resolution is True
    # Synthetic negative categories are enabled by default.
    assert args.synthetic_pmhc_negative_ratio > 0.0
    assert args.synthetic_class_i_no_mhc_beta_negative_ratio > 0.0
    assert args.synthetic_processing_negative_ratio > 0.0

def test_parser_wires_train_synthetic_run_dir():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "synthetic",
            "--run-dir",
            "/tmp/synth",
        ]
    )
    assert args.func is train_cli.cmd_train_synthetic
    assert args.run_dir == "/tmp/synth"


def test_parser_wires_train_config_file():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "synthetic",
            "--config",
            "train.yaml",
        ]
    )
    assert args.func is train_cli.cmd_train_synthetic
    assert args.config == "train.yaml"


def test_parser_wires_synthetic_advanced_flags():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "synthetic",
            "--weight_decay",
            "0.02",
            "--no-uncertainty-weighting",
            "--use-pcgrad",
        ]
    )
    assert args.func is train_cli.cmd_train_synthetic
    assert args.weight_decay == 0.02
    assert args.use_uncertainty_weighting is False
    assert args.use_pcgrad is True


def test_parser_accepts_supervised_loss_aggregation_flags():
    parser = create_parser()
    synth = parser.parse_args(
        [
            "train",
            "synthetic",
            "--supervised-loss-aggregation",
            "task_mean",
        ]
    )
    assert synth.func is train_cli.cmd_train_synthetic
    assert synth.supervised_loss_aggregation == "task_mean"

    unified = parser.parse_args(
        [
            "train",
            "unified",
            "--supervised-loss-aggregation",
            "task_mean",
        ]
    )
    assert unified.func is train_cli.cmd_train_unified
    assert unified.supervised_loss_aggregation == "task_mean"


def test_parser_wires_consistency_and_pmhc_flags():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--synthetic-pmhc-negative-ratio",
            "0.9",
            "--consistency-cascade-weight",
            "0.3",
            "--consistency-no-b2m-weight",
            "1.2",
            "--consistency-tcell-context-weight",
            "0.4",
        ]
    )
    assert args.func is train_cli.cmd_train_unified
    assert args.synthetic_pmhc_negative_ratio == 0.9
    assert args.consistency_cascade_weight == 0.3
    assert args.consistency_no_b2m_weight == 1.2
    assert args.consistency_tcell_context_weight == 0.4


def test_parser_train_unified_accepts_no_mhc_beta_negative_ratio_flag():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--synthetic-class-i-no-mhc-beta-negative-ratio",
            "0.33",
        ]
    )
    assert args.func is train_cli.cmd_train_unified
    assert args.synthetic_class_i_no_mhc_beta_negative_ratio == 0.33


def test_parser_train_iedb_alias_still_supported():
    parser = create_parser()
    args = parser.parse_args(["train", "iedb"])
    assert args.func is train_cli.cmd_train_unified


def test_parser_train_unified_accepts_canary_profile():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--profile", "canary"])
    assert args.func is train_cli.cmd_train_unified
    assert args.profile == "canary"


def test_parser_train_unified_accepts_diagnostic_profile():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--profile", "diagnostic"])
    assert args.func is train_cli.cmd_train_unified
    assert args.profile == "diagnostic"


def test_parser_train_unified_allows_raw_fallback():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--allow-raw-fallback"])
    assert args.func is train_cli.cmd_train_unified
    assert args.require_merged_input is False


def test_parser_train_unified_allows_unresolved_mhc_override():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--allow-unresolved-mhc"])
    assert args.func is train_cli.cmd_train_unified
    assert args.strict_mhc_resolution is False


def test_parser_train_unified_filter_unresolved_mhc_flag():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--filter-unresolved-mhc"])
    assert args.func is train_cli.cmd_train_unified
    assert args.filter_unresolved_mhc is True


def test_parser_train_unified_no_filter_unresolved_mhc_flag():
    parser = create_parser()
    args = parser.parse_args(["train", "unified", "--no-filter-unresolved-mhc"])
    assert args.func is train_cli.cmd_train_unified
    assert args.filter_unresolved_mhc is False


def test_parser_train_unified_accepts_mhc_attention_sparsity_flags():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--mhc-attention-sparsity-weight",
            "0.2",
            "--mhc-attention-sparsity-min-residues",
            "35",
            "--mhc-attention-sparsity-max-residues",
            "55",
        ]
    )
    assert args.func is train_cli.cmd_train_unified
    assert args.mhc_attention_sparsity_weight == 0.2
    assert args.mhc_attention_sparsity_min_residues == 35.0
    assert args.mhc_attention_sparsity_max_residues == 55.0


def test_parser_train_unified_probe_tracking_defaults_and_overrides():
    parser = create_parser()
    defaults = parser.parse_args(["train", "unified"])
    assert defaults.track_probe_affinity is True
    assert defaults.probe_peptide == "SLLQHLIGL"
    assert defaults.probe_alleles == "HLA-A*02:01,HLA-A*24:02"

    disabled = parser.parse_args(
        [
            "train",
            "unified",
            "--no-track-probe-affinity",
            "--probe-peptide",
            "SIINFEKL",
            "--probe-alleles",
            "HLA-A*02:01",
        ]
    )
    assert disabled.track_probe_affinity is False
    assert disabled.probe_peptide == "SIINFEKL"
    assert disabled.probe_alleles == "HLA-A*02:01"


def test_parser_train_unified_performance_and_loader_flags():
    parser = create_parser()
    args = parser.parse_args(
        [
            "train",
            "unified",
            "--num-workers",
            "2",
            "--no-pin-memory",
            "--no-profile-performance",
            "--perf-log-interval-batches",
            "25",
        ]
    )
    assert args.func is train_cli.cmd_train_unified
    assert args.num_workers == 2
    assert args.pin_memory is False
    assert args.profile_performance is False
    assert args.perf_log_interval_batches == 25


def test_parser_train_unified_pmhc_flow_tracking_defaults_and_overrides():
    parser = create_parser()
    defaults = parser.parse_args(["train", "unified"])
    assert defaults.track_pmhc_flow is True
    assert defaults.pmhc_flow_batches == 2
    assert defaults.pmhc_flow_max_samples == 512

    disabled = parser.parse_args(
        [
            "train",
            "unified",
            "--no-track-pmhc-flow",
            "--pmhc-flow-batches",
            "3",
            "--pmhc-flow-max-samples",
            "120",
        ]
    )
    assert disabled.track_pmhc_flow is False
    assert disabled.pmhc_flow_batches == 3
    assert disabled.pmhc_flow_max_samples == 120


def test_parser_train_unified_output_latent_tracking_defaults_and_overrides():
    parser = create_parser()
    defaults = parser.parse_args(["train", "unified"])
    assert defaults.track_output_latent_stats is True
    assert defaults.output_latent_stats_batches == 2
    assert defaults.output_latent_stats_max_samples == 512

    disabled = parser.parse_args(
        [
            "train",
            "unified",
            "--no-track-output-latent-stats",
            "--output-latent-stats-batches",
            "5",
            "--output-latent-stats-max-samples",
            "200",
        ]
    )
    assert disabled.track_output_latent_stats is False
    assert disabled.output_latent_stats_batches == 5
    assert disabled.output_latent_stats_max_samples == 200
