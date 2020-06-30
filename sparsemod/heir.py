# -*- coding: utf-8 -*-
from os import listdir, remove, path
from numpy import arange, any as npany, sum as npsum, zeros
from torch import Size


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = npany(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = npsum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def clean_models_return_pareto(data, folder):
    """
    Deletes the models which do not pertain to the pareto front
    """
    data_use = data.df.pivot(columns="metric_name", values="mean", index="arm_name")
    pareto_mask = is_pareto_efficient(data_use.values)
    pareto_arms = data_use.index[pareto_mask].values

    # TODO: careful with batch in ax
    for file_name in listdir(folder):
        if not ("_").join(file_name.split(".")[0].split("_")[:2]) in set(pareto_arms):
            remove(path.join(folder, file_name))
    return pareto_arms


def copy_weights(old_model, net):
    """
    Tolerates only copies until tensors of 4 dims
    """
    for name_old, weight_old in old_model.state_dict().items():
        for name_new, weight_new in net.state_dict().items():
            if name_old == name_new:
                length = len(weight_new.shape)
                copy_shape = return_shapes(weight_old, weight_new)
                if length == 1:
                    net.state_dict()[name_new][: copy_shape[0]] = weight_old[
                        : copy_shape[0]
                    ]
                elif length == 2:
                    net.state_dict()[name_new][
                        : copy_shape[0], : copy_shape[1]
                    ] = weight_old[: copy_shape[0], : copy_shape[1]]
                elif length == 3:
                    net.state_dict()[name_new][
                        : copy_shape[0], : copy_shape[1], : copy_shape[2]
                    ] = weight_old[: copy_shape[0], : copy_shape[1], : copy_shape[2]]
                elif length == 4:
                    net.state_dict()[name_new][
                        : copy_shape[0],
                        : copy_shape[1],
                        : copy_shape[2],
                        : copy_shape[3],
                    ] = weight_old[
                        : copy_shape[0],
                        : copy_shape[1],
                        : copy_shape[2],
                        : copy_shape[3],
                    ]
                else:
                    net.state_dict()[name_new] = weight_old
    return net


def return_shapes(weight_old, weight_new):
    """Returns the sahpe to maintain, ie. the bigger """
    return Size([i if j > i else j for i, j in zip(weight_old.shape, weight_new.shape)])
