"""
Parser functions for HMC datasets.
"""

import logging

# Criar um logger
logger = logging.getLogger(__name__)


def create_example(data):
    """
    Creates a dictionary of tensors from input data.

    Args:
        data: A tuple containing features and labels.

    Returns:
        A dictionary containing the features and labels as tensors.
    """
    features, labels = data
    example_tensor = {"features": features, "labels": labels}

    return example_tensor
