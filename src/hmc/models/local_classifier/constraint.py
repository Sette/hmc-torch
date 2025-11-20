from typing import List, Optional


from hmc.models.local_classifier.baseline import HMCLocalModel


# ============================================================================
# CHILD CLASS 2 - Local Classification Model constraint_old
# ============================================================================


class HMCLocalModelConstraint(HMCLocalModel):
    """
    Hierarchical model where each level makes independent predictions.

    Child class that implements local classification at each level.
    Optionally supports residual connections between levels.
    This variant includes hierarchical constraints specific to the model.
    """

    def __init__(
        self,
        levels_size: List[int],
        input_size: int,
        hidden_dims: List[int],
        results_path: str,
        num_layers: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        active_levels: Optional[List[int]] = None,
        residual: bool = False,
    ):
        """
        Initialize local classification model with constraints.

        Args:
            levels_size: Number of classes at each level
            input_size: Dimension of input features
            hidden_dims: Hidden layer sizes for each level
            results_path: Path to save model checkpoints
            num_layers: Number of layers for each level (default: 2)
            dropouts: Dropout rates for each level (default: 0.0)
            active_levels: Indices of levels to train
            residual: Whether to use residual connections
        """
        super(HMCLocalModelConstraint, self).__init__(
            levels_size=levels_size,
            input_size=input_size,
            hidden_dims=hidden_dims,
            results_path=results_path,
            num_layers=num_layers,
            dropouts=dropouts,
            active_levels=active_levels,
            residual=residual,
        )
