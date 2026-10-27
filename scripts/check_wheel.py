"""Smoke-test a built wheel outside the checkout, reusing installed dependencies."""

import argparse
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import venv


def check_wheel(wheel):
    wheel = Path(wheel).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="presto-wheel-") as directory:
        root = Path(directory)
        environment = root / "environment"
        # CI already installed the tested dependency set. Keep this package's
        # files in a separate environment so editable source cannot mask a
        # missing module or runtime resource in the wheel.
        # POSIX standalone Python builds may locate libpython relative to their
        # executable; copying that executable alone breaks the new interpreter.
        venv.EnvBuilder(with_pip=True, system_site_packages=True, symlinks=os.name != "nt").create(
            environment
        )
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        # A nested venv inherits base-Python packages, not its caller's venv.
        # Append the caller's dependency paths; this wheel's own package remains
        # first, and the import-origin assertion below rejects an editable copy.
        target_site = Path(
            subprocess.check_output(
                [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
                env=env,
            ).strip()
        )
        dependency_sites = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
        (target_site / "presto_smoke_dependencies.pth").write_text(
            "".join(f"import site; site.addsitedir({path!r})\n" for path in dependency_sites)
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--ignore-installed",
                str(wheel),
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
import importlib.metadata
from pathlib import Path
import sys
import presto
assert Path(presto.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert presto.__version__ == importlib.metadata.version('presto')
from presto.data.allele_resolver import HUMAN_B2M_SEQUENCE, MOUSE_B2M_SEQUENCE
assert HUMAN_B2M_SEQUENCE and MOUSE_B2M_SEQUENCE
from presto.data.bulk_ms import records_from_bulk_frame
from presto.data.loaders import PrestoDataset
from presto.data.collate import PrestoCollator
records, stats = records_from_bulk_frame([
    {'peptide': 'ACDEFGHIK', 'digestion_enzyme': 'Trypsin/P'}
])
assert stats['n_detectability_labels'] == 0
assert all(record.detectability_label is None for record in records)
batch = PrestoCollator()(PrestoDataset(bulk_ms_records=records).samples)
assert 'ms_detectability' not in batch.target_masks
assert batch.target_masks['excision'].all()
print('Installed wheel version, B2M resources, bulk loader and masks verified.')
""",
            ],
            cwd=root,
            env=env,
            check=True,
        )
        subprocess.run([str(python), "-I", "-m", "presto", "--help"], cwd=root, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check_wheel(parser.parse_args().wheel)
