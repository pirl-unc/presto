"""Binding affinity prediction heads — registry and imports."""

from .base import AffinityHead
from .regression import MHCflurryHead, LogMSEHead
from .twohot import TwoHotHead
from .hlgauss import HLGaussHead
from .gaussian import GaussianHead
from .quantile import QuantileHead

HEAD_REGISTRY = {
    "mhcflurry": MHCflurryHead,
    "log_mse": LogMSEHead,
    "twohot": TwoHotHead,
    "hlgauss": HLGaussHead,
    "gaussian": GaussianHead,
    "quantile": QuantileHead,
}

__all__ = [
    "AffinityHead",
    "MHCflurryHead",
    "LogMSEHead",
    "TwoHotHead",
    "HLGaussHead",
    "GaussianHead",
    "QuantileHead",
    "HEAD_REGISTRY",
]
