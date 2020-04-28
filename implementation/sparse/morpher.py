# -*- coding: utf-8 -*-

from numpy.random import choice, randint, random


class Morpher(object):
    """
    Retrieves configurations of trials and adds morphisms to them.
    """

    def __init__(self, operations):
        self.operations = operations

    def retrieve_best_configurations(self, experiment, pareto_arms):
        self.configs = {}
        for arm in pareto_arms:
            params = experiment.arms_by_name[arm].parameters
            self.configs[arm] = params

    def apply_morphs(self, n_morphs=5, max_n_changes=10):
        configs_sub = choice(list(self.configs), size=n_morphs)
        n_changes_x_morph = randint(1, max_n_changes, size=n_morphs)
        new_configs = {}
        for i, (config, changes) in enumerate(zip(configs_sub, n_changes_x_morph)):
            ops = choice(list(self.operations), changes)
            new_configs[(str(i), config)] = self.configs[config]
            for k in ops:
                new_configs[(str(i), config)] = self.operations[k](
                    new_configs[(str(i), config)]
                )
        return new_configs


