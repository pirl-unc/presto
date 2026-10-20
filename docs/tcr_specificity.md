# Paired TCR–pMHC functional specificity

`TCRSpecificityModel` is an optional wrapper around an existing `Presto` model.
It takes paired alpha/beta amino-acid chains plus the existing peptide/MHC
inputs and returns a pair-specific functional-response logit. Separate
position-aware chain encoders feed a joint receptor/pMHC response head.
To score a candidate pMHC panel, repeat the receptor pair for each candidate
and keep the batches aligned. The wrapper does not generate an unknown epitope.

This implementation ships **no trained TCR weights or validated accuracy**.
Its sigmoid output is an uncalibrated score. It is neither molecular TCR–pMHC
KD nor evidence that a receptor recognizes every peptide absent from the
training set. Canonical Presto's repertoire recognition and pMHC-only
`tcr_evidence` outputs retain their existing meanings.

## Intended data: TCRvdb

The requested resource is the independent functional re-testing in
[Messemaker et al. (PMCID PMC12132471)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12132471/),
not a VDJdb confidence-score filter. The
[author repository](https://github.com/schumacherlab/TCRvdb) lists an October 1,
2026 dataset and the subsequent publication DOI
[10.1016/j.immuni.2026.09.007](https://doi.org/10.1016/j.immuni.2026.09.007).
As checked on October 20, the repository directs users to request access and
requires permission for redistribution.

No complete author CSV has been acquired or inspected for this implementation.
GitHub searches found analysis scripts and a small derived non-validating
subset, which is not an adequate replacement. The loader below deliberately
uses a **Presto interchange schema**, not guessed author column names.
An author-file adapter and training cohort must wait for the actual authorized
release. [Hitlist #722](https://github.com/pirl-unc/hitlist/issues/722) tracks
that upstream contract.

Full supplied chains are encoded without truncation (default limit 512 residues
per chain). Record whether sequences were observed or reconstructed, and retain
the reference/reconstruction procedure. A CDR3 plus V/J assignment does not
establish an observed full-length sequence; length alone cannot verify this.
Never infer paired chains merely because they share an epitope.

## Python interface

Use a separate backbone instance when freezing it: the wrapper changes that
instance's gradient and training state. The default freezes Presto, including
dropout. Explicit joint fine-tuning is available through
`set_backbone_frozen(False)`. Chain encoders and the scorer remain trainable.

```python
import torch

from presto.models import TCRSpecificityModel, functional_response_loss
from presto.training.checkpointing import load_model_from_checkpoint

backbone, _ = load_model_from_checkpoint("pmhc_checkpoint.pt")
model = TCRSpecificityModel(backbone)

# Paired chains and pMHC tensors must have the same batch ordering.
# pmhc_batch is an existing PrestoBatch prepared by PrestoCollator from
# resolved groove segments, with no receptor labels added to model_inputs().
outputs = model(
    tcr_alpha=alpha_amino_acid_chains,
    tcr_beta=beta_amino_acid_chains,
    **pmhc_batch.model_inputs(),
)
loss = functional_response_loss(outputs["tcr_functional_logit"], measured_targets)
loss.backward()

# Save both architectures and states; include the qualified assay/data scope.
model.save_checkpoint(
    "specificity.pt",
    metadata={"source_sha256": source_hash, "assay_scope": assay_scope},
)
restored, metadata = TCRSpecificityModel.load_checkpoint("specificity.pt")
with torch.no_grad():
    scores = restored(
        tcr_alpha=alpha_amino_acid_chains,
        tcr_beta=beta_amino_acid_chains,
        **pmhc_batch.model_inputs(),
    )["tcr_functional_score"]
```

Move the entire wrapper and pMHC batch to the same device. Raw chains are
tokenized on the wrapper's device. Checkpoint metadata should contain ordinary
strings/numbers/lists/dictionaries, compatible with Torch's weights-only loader.
The wrapper checkpoint format is separate from canonical Presto checkpoints.

## Normalized local CSV contract

`presto.data.tcr_specificity.load_tcr_specificity_csv(path)` returns admitted
records, the source SHA-256 and exclusion counts. Required columns are:

| Fields | Meaning |
| --- | --- |
| `record_id`, `study_id`, `assay_id` | Stable source observation, study and functional assay scope |
| `peptide`, `mhc_a`, `mhc_b`, `mhc_class` | Tested pMHC restriction; MHC names must be resolved to groove inputs before inference; class II requires both chains |
| `tcr_alpha`, `tcr_beta` | Paired canonical amino-acid chains |
| `sequence_provenance`, `sequence_reference` | `observed_full_length` or `reconstructed`, with source or reconstruction provenance |
| `outcome` | `validated`, `not_validated`, `indeterminate`, or `unassayed` |
| `quality` | `pass`, `fail`, or `unknown` under the declared source policy |

Only QC-passing `validated`/`not_validated` observations enter the binary loss.
The latter means a tested pair did not validate under that assay's conditions;
it is not a universal molecular non-binder. Failed QC, indeterminate and unassayed
records are excluded and counted. The source adapter must retain raw readouts,
statistics, QC definitions and author thresholds in a companion provenance
bundle. Source IDs, QC and outcomes never enter the predictive features.

Do not turn unobserved combinations or shuffled receptor/peptide pairs into
measured negatives. Do not discard measured non-validation merely because the
resource is called “validated.” Keep repeated measurements together and audit
conflicting outcomes before choosing aggregation.

Before reporting accuracy, freeze receptor-cluster, epitope/MHC and source-study
split policies. Report discrimination and calibration within the assayed
coverage, with explicit extrapolation limits. A small number of tested pMHCs
cannot establish performance on arbitrary unseen epitopes. Keep restricted
records and per-record outputs out of public artifacts unless permitted.
