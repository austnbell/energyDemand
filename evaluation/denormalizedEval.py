# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:23:54 2020

Function that denormalizes our data and brings it up to the original values 

@author: Austin Bell
"""

import pandas as pd
import numpy as np


# return our load values to their initial form 
def denormalizeLoad(load, node_min_max):
    seq_by_nodes = np.split(load, load.shape[1], axis = 1)

    denormalized_arrays = []
    for i, node_seq in enumerate(seq_by_nodes):
        node_seq = np.squeeze(node_seq)
        min_load = node_min_max.loc[node_min_max.index == i, "min_load"].values[0]
        max_load = node_min_max.loc[node_min_max.index == i, "max_load"].values[0]
        denormalize = node_seq * (max_load - min_load) + min_load
        denormalized_arrays.append(denormalize)

    denormalized_arrays = np.stack(denormalized_arrays,1)
    return denormalized_arrays