# Repair the clean-CI development dependency contract

2026-10-27, Codex / GPT-6. Published e04dfcf CI40432992416 completed with docs
passing and test job121954923439 failing:2664passed,3skipped,3fixture errors.
All three errors import the resource-guarded censor experiment helper and lack
`psutil`. The shared local environment already contains psutil, explaining the
passing local full suite. No scientific test assertion failed.

Declare psutil in the existing development extra and install that same extra in
CI with Hitlist. Keep CPU Torch installation, pinned Ruff, test coverage and all
frozen experiment sources unchanged. Do not add skips or fake the resource guard.
No shared-environment install or editable repointing is needed.

Verify the affected tests, repository lint/format and wheel metadata/build using
existing tools. The wheel must declare psutil only for the development extra,
not as a mandatory inference dependency. Push the tested change and require
final-head CI, including its full clean-environment test/build steps, to finish.
Preserve the failed CI log and its exact head. This metadata/workflow change
does not modify the running comparator's numerical runtime or installed packages.

## Local wheel-check failure and revised verification

The original check and one unsandboxed retry both failed before package installation:
the copied uv-managed macOS Python cannot locate its relative libpython dylib.
A temporary-environment diagnostic reproduces exit -6 with copies and succeeds
with symlinks. This is not a sandbox denial or a wheel import failure. Preserve
both failed logs and the diagnostic. Make the existing smoke helper use POSIX
interpreter symlinks (retain Windows copies), then repeat the exact offline wheel
check. Do not change model code, scientific runners or the shared environment.

With interpreter startup repaired, installation succeeds but the nested venv
cannot import Torch: system_site_packages exposes the base Python's packages,
not the calling shared virtual environment's dependencies. Explicitly add the
caller's site-packages to the temporary environment, after its own packages.
Retain the assertion that Presto itself resolves inside the temporary environment,
so editable source cannot hide wheel omissions. Verify via the actual smoke command.
