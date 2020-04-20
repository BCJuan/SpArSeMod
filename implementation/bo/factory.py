# -*- coding: utf-8 -*-
from logging import Logger
from typing import Dict, List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.registry import (
    Cont_X_trans,
    Models,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch import (
    TAcqfConstructor,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)

from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast

# MOD
from .botorch_defaults import (
    get_NEI,
    predict_from_model,
    scipy_optimizer,
    get_and_fit_model,
)

#
logger: Logger = get_logger(__name__)


DEFAULT_TORCH_DEVICE = torch.device("cpu")


def get_botorch(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    model_constructor: TModelConstructor = get_and_fit_model,
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_NEI,  # pyre-ignore[9]
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    refit_on_update: bool = True,
    optimization_config: Optional[OptimizationConfig] = None,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("`BotorchModel` requires non-empty data.")
    return checked_cast(
        TorchModelBridge,
        Models.BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            optimization_config=optimization_config,
        ),
    )
