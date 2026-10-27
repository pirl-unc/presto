"""Compatibility launcher: all new training uses the shared observation manifest.

Historical flags/recipes live at Git commit 9dd71348, not in a selectable backend.
"""

from presto.cli.train import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
