"""Historical paired-TCR checkpoint loading; new TCR fitting uses Presto."""

from ..legacy.tcr_specificity import validate_chain


class TCRSpecificityModel:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Separate TCR models are retired; use PrestoConfig endpoints "
            "tcr_binding/tcr_response and the observation-manifest trainer"
        )

    @classmethod
    def load_checkpoint(cls, path, *, map_location="cpu"):
        from ..legacy.tcr_specificity import TCRSpecificityModel as HistoricalModel

        return HistoricalModel.load_checkpoint(path, map_location=map_location)


__all__ = ["TCRSpecificityModel", "validate_chain"]
