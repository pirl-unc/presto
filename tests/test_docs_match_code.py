"""Docs must not describe things the code does not do.

Three kinds of drift were found by hand in this codebase and each would have
been caught here: a contract claiming an embedding row had zero corpus support
after it had gained 21,394 rows; a DAG equation still written in terms of
`liberation` after the rename to `excision`; and a design section specifying
`core_context_vec` in full, with a formula, for a component that was never
built.

The check is deliberately narrow. It does not try to validate prose -- it
verifies that identifiers the docs present as *model outputs* actually appear
in `forward()`, and that names retired by a rename are gone.
"""

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from presto.models.presto import Presto  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"

#: Documented but deliberately absent, each with the reason it is exempt.
KNOWN_NOT_OUTPUTS = {
    # design.md says in prose that these do not exist ("There is no pooled
    # peptide_vec or combined mhc_vec"); the backticks are the doc denying them.
    "peptide_vec": "design.md explicitly states it does not exist",
    "mhc_vec": "design.md explicitly states it does not exist",
    # Specified in design.md S6.3 and never implemented. The section carries a
    # [specified, not built] banner.
    "core_context_vec": "S6.3 is marked as not built",
    "context_vec": "an argument name, not an output",
    # Abandoned subsystem; docs/tcr_spec.md carries an abandoned banner.
    "tcr_vec": "models/tcr.py is abandoned, see tcr_spec.md",
    "match_logit": "models/tcr.py is abandoned, see tcr_spec.md",
    # Internal latents, reachable via latent_vecs / trunk_state rather than as
    # top-level output keys.
    "binding_affinity_vec": "trunk latent",
    "binding_stability_vec": "trunk latent",
    "processing_class1_vec": "trunk latent",
    "processing_class2_vec": "trunk latent",
}

#: Identifiers retired by a rename. Their reappearance in docs is drift.
RETIRED_NAMES = {
    "liberation": "renamed to `excision`",
    "length_score_value": "renamed to `length_preference`",
    "processing_condition_embed": "deleted; cellular state is not a trunk token",
    "ifn_ab": "renamed to `ifn_type1`",
    "apc_cell_class": (
        "replaced by the cell_lineage / sample_origin / disease_state axes; "
        "it was built from a 58.3%-covered field and conflated orthogonal axes"
    ),
}


@pytest.fixture(scope="module")
def output_keys():
    torch.manual_seed(0)
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    model.eval()
    with torch.no_grad():
        out = model(
            pep_tok=torch.randint(4, 24, (2, 10)),
            mhc_a_tok=torch.randint(4, 24, (2, 40)),
            mhc_b_tok=torch.randint(4, 24, (2, 40)),
            mhc_class="I",
        )
    keys = set(out)
    keys |= set(out.get("latent_vecs", {}) or {})
    return keys


def _documented_output_identifiers():
    pattern = re.compile(
        r"`([a-z][a-z0-9_]*(?:_logit|_logits|_prob|_probs|_vec|_score"
        r"|_panel_[a-z_]+))`"
    )
    found = {}
    for path in sorted(DOCS.glob("*.md")):
        for name in pattern.findall(path.read_text()):
            found.setdefault(name, set()).add(path.name)
    return found


class TestDocumentedOutputsExist:
    def test_every_documented_output_is_produced(self, output_keys):
        documented = _documented_output_identifiers()
        missing = {
            name: sorted(files)
            for name, files in documented.items()
            if name not in output_keys and name not in KNOWN_NOT_OUTPUTS
        }
        assert missing == {}, (
            "these are documented as model outputs but forward() does not "
            f"produce them: {missing}. Either build them, or add them to "
            "KNOWN_NOT_OUTPUTS with the reason."
        )

    def test_exemptions_are_still_needed(self, output_keys):
        """An exemption that starts existing should lose its exemption."""
        stale = sorted(name for name in KNOWN_NOT_OUTPUTS if name in output_keys)
        assert stale == [], f"these now exist and should leave KNOWN_NOT_OUTPUTS: {stale}"


class TestRetiredNamesAreGone:
    @pytest.mark.parametrize("name,reason", sorted(RETIRED_NAMES.items()))
    def test_no_doc_still_uses_the_old_name(self, name, reason):
        offenders = []
        for path in sorted(DOCS.glob("*.md")):
            text = path.read_text()
            for line_no, line in enumerate(text.splitlines(), start=1):
                if name in line and "renamed" not in line and "earlier name" not in line:
                    # A line explaining the retirement is allowed to name it.
                    if any(
                        marker in line for marker in ("no longer", "used to", "deleted", "replace")
                    ):
                        continue
                    offenders.append(f"{path.name}:{line_no}")
        assert offenders == [], f"`{name}` ({reason}) still appears in {offenders}"
