"""Manifest-only compatibility launcher for the shared Presto trainer."""

from presto.cli.train import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
