"""One manifest training entry point for every task, source and launch alias."""

import argparse

from .training_sources import add_training_source_arguments


def run(args):
    from ..training.modular_run import run_manifest

    return run_manifest(args)


def cmd_train(args):
    run(args)
    return 0


# Existing command names are aliases, with no model/trainer/settings of their own.
cmd_train_unified = cmd_train
cmd_train_synthetic = cmd_train
cmd_train_iedb = cmd_train


def main(argv=None):
    from ..training.config_io import explicit_cli_destinations
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    add_training_source_arguments(parser)
    supplied = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(supplied)
    args._explicit_cli_dests = explicit_cli_destinations(parser, supplied, args)
    return run(args)


if __name__ == "__main__":
    main()
