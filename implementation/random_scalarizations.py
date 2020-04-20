#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:17:55 2019

@author: kostal
"""

from scipy import random
from numpy import log


class random_scalarizer(object):
    def __init__(
        self, weights, accuracy, ram, latency, top_w=10**8, top_r=10**8, top_l=10**4, objectives=3
    ):
        self.weights = log(weights) / log(top_w)
        self.accuracy = accuracy
        self.ram = log(ram) / log(top_r)
        self.latency = log(latency)/log(top_l)
        self.objectives = objectives

    def sampler(self):
        w_list = [
            random.uniform(self.accuracy - 0.025, self.accuracy + 0.025),
            random.uniform(self.weights - 0.025, self.weights + 0.025),
            random.uniform(self.ram - 0.025, self.ram + 0.025),
            random.uniform(self.latency - 0.025, self.latency + 0.025),
        ]
        print(w_list)
        return w_list[: self.objectives]
