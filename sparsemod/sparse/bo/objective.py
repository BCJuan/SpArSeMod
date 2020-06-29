r"""
Objective Modules to be used with acquisition functions.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import torch
from botorch.utils import apply_constraints
from torch import Tensor
from torch.nn import Module
from botorch.acquisition.objective import MCAcquisitionObjective


class MaxMCObjective(MCAcquisitionObjective):

    r""" Maximum of scalarized objectives
    If scalarized weights are w_i*obj_i it selects the maximum
    w_i*obj_i
    """

    def __init__(self, weights: Tensor) -> None:

        r"""Max Objective.
        Args:
            weights: A one-dimensional tensor with `o` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor) -> Tensor:

        r"""Evaluate the linear objective on the samples.
        Args:
            samples: A `sample_shape x batch_shape x q x o`-dim tensors of
                samples from a model posterior.
        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """

        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        num_obj = samples.shape[-1]
        max_weighted = [
            torch.matmul(
                torch.index_select(samples, len(samples.shape) - 1, torch.tensor([i])),
                torch.DoubleTensor([self.weights[i]]),
            )
            for i in range(num_obj)
        ]
        max_weighted, indices = torch.max(torch.stack(max_weighted), dim=0)
        return max_weighted
