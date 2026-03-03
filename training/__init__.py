"""Training utilities for Presto.

Exports canonical unified-training components.
"""

from .config import (
    Config,
    OptimizerConfig,
    LRSchedule,
    default_config,
    muon_config,
    fast_config,
    binding_only_config,
)

from .losses import (
    censor_aware_loss,
    mil_bag_loss,
    UncertaintyWeighting,
    CombinedLoss,
    safe_bce_with_logits,
    focal_loss,
)
from .checkpointing import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    MODEL_CLASS,
    build_model_config,
    infer_model_config_from_state_dict,
    save_model_checkpoint,
    load_model_from_checkpoint,
)
from .trainer import Trainer
from .tasks import (
    # Core classes
    Task,
    TaskSpec,
    TASK_REGISTRY,
    get_task,
    register_task,
    route_sample,
    # MHC parsing (using mhcgnomes when available)
    parse_mhc_allele,
    generate_shuffled_negatives,
    generate_random_negatives,
    # Chain classification
    MHCChainTypeTask,
    ReceptorChainTypeTask,
    SpeciesTask,
    MHC_CHAIN_TYPE_MAP,
    RECEPTOR_CHAIN_TYPE_MAP,
    SPECIES_MAP,
    # Pairing
    MHCPairingTask,
    TCRPairingTask,
    MHC_VALID_PAIRINGS,
    # Binding and presentation support
    BindingTask,
    ElutionTask,
    ProcessingTask,
    StabilityTask,
    # Recognition and immunogenicity
    TCRpMHCMatchingTask,
    ImmunogenicityTask,
    TcellAssayTask,
    # Task balancing
    TaskBalancer,
    StaticBalancer,
    UncertaintyBalancer,
    RandomBalancer,
    TaskTracker,
)
