"""Neural network models."""

from .encoders import SequenceEncoder, ProjectionHead, l2_normalize
from .pmhc import (
    ProcessingModule,
    PMHCEncoder,
    BindingModule,
    StableBindingHead,
    PresentationBottleneck,
    stable_noisy_or,
    posterior_attribution,
    enumerate_core_windows,
)
from .tcr import (
    TCREncoder,
    ChainClassifier,
    ChainAttributeClassifier,
    CellTypeClassifier,
    TCRpMHCMatcher,
    RepertoireHead,
    get_compatibility_mask,
    info_nce_loss,
)
from .heads import (
    AssayHeads,
    TCellHead,
    ElutionHead,
    to_log10_nM,
    from_log10_nM,
    normalize_tm,
    denormalize_tm,
)
from .affinity import (
    DEFAULT_MIN_AFFINITY_NM,
    DEFAULT_MAX_AFFINITY_NM,
    DEFAULT_BINDING_MIDPOINT_NM,
    DEFAULT_BINDING_LOG10_SCALE,
    max_log10_nM,
    affinity_nm_to_log10,
    affinity_log10_to_nm,
    binding_logit_from_kd_log10,
    binding_prob_from_kd_log10,
    normalize_binding_target_log10,
)
from .presto import Presto
