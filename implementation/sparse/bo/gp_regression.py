# -*- coding: utf-8 -*-

r"""
Gaussian Process Regression models based on GPyTorch models.
"""

from typing import Any, List, Optional, Union


from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors.torch_priors import GammaPrior

# MOD
from torch import Tensor
from botorch import settings
from botorch.sampling.samplers import MCSampler
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.kernels import ProductKernel
from .arc_kernel import ArcKernel

#

MIN_INFERRED_NOISE_LEVEL = 1e-4


class FixedNoiseGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    r"""A single-task exact GP model using fixed noise levels.

    A single-task exact GP that uses fixed observation noise levels. This model
    also uses relatively strong priors on the Kernel hyperparameters, which work
    best when covariates are normalized to the unit cube and outcomes are
    standardized (zero mean, unit variance).

    This model works in batch mode (each batch having its own hyperparameters).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        r"""A single-task exact GP model using fixed noise levels.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: A `batch_shape x n x m` tensor of observed measurement
                noise.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).

        Example:
            >>> train_X = torch.rand(20, 2)
            >>> train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
            >>> train_Yvar = torch.full_like(train_Y, 0.2)
            >>> model = FixedNoiseGP(train_X, train_Y, train_Yvar)
        """
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        validate_input_scaling(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, train_Yvar = self._transform_tensor_args(
            X=train_X, Y=train_Y, Yvar=train_Yvar
        )
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_Yvar, batch_shape=self._aug_batch_shape
        )
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        # MOD####
        base_kernel = MaternKernel(nu=2.5)
        self.covar_module = ProductKernel(ScaleKernel(
            ArcKernel(
                base_kernel,
                angle_prior=GammaPrior(0.5, 1),
                radius_prior=GammaPrior(3, 2),
                ard_num_dims=train_X.shape[-1],
                batch_shape=self._aug_batch_shape,
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
            batch_shape=self._aug_batch_shape,
        ))

        ######
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        # self._subset_batch_dict = {
        #     "mean_module.constant": -2,
        #     "covar_module.raw_outputscale": -1,
        #     "covar_module.base_kernel.raw_lengthscale": -3,
        # }
        self.to(train_X)

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Union[bool, Tensor] = True,
        **kwargs: Any,
    ) -> "FixedNoiseGP":
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X` (if `observation_noise=True`,
        this includes observation noise taken as the mean across the observation
        noise in the training data. If `observation_noise` is a Tensor, use
        it directly as the observation noise to add).
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: If True, include the mean across the observation
                noise in the training data as observation noise in the posterior
                from which the samples are drawn. If a Tensor, use it directly
                as the specified measurement noise.

        Returns:
            The constructed fantasy model.
        """
        propagate_grads = kwargs.pop("propagate_grads", False)
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X, observation_noise=observation_noise, **kwargs)
        Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
        # Use the mean of the previous noise values (TODO: be smarter here).
        # noise should be batch_shape x q x m when X is batch_shape x q x d, and
        # Y_fantasized is num_fantasies x batch_shape x q x m.
        noise_shape = Y_fantasized.shape[1:]
        noise = self.likelihood.noise.mean().expand(noise_shape)
        return self.condition_on_observations(X=X, Y=Y_fantasized, noise=noise)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def subset_output(self, idcs: List[int]) -> "BatchedMultiOutputGPyTorchModel":
        r"""Subset the model along the output dimension.

        Args:
            idcs: The output indices to subset the model to.

        Returns:
            The current model, subset to the specified output indices.
        """
        new_model = super().subset_output(idcs=idcs)
        full_noise = new_model.likelihood.noise_covar.noise
        new_noise = full_noise[..., idcs if len(idcs) > 1 else idcs[0], :]
        new_model.likelihood.noise_covar.noise = new_noise
        return new_model
