# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:22:56 2020

@author: Austin Bell
"""

import os, sys
import pandas as pd
import numpy as np
from datetime import datetime
import time
import re
import importlib

import torch
import torch.nn as nn

# user functions
import sys
sys.path.append("..")
from dataUtils import loadEnergyData, processData, energyDataset, getDatasets, normalizeAdjMat
from modelUtils import dotDict

##### Load our args
config_file = "baseSTGCN_config"
c = importlib.import_module("configs."+config_file)
args = c.args

print(args)



torch.manual_seed(0)
np.random.seed(0)

# last three months - will add test set later
validation_range = ["2014-10-01 00:00:00", "2014-12-31 23:00:00"]
validation_range = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in validation_range]



# we must have these arguments regardless of actual config
args.save_seq = True
args.load_seq = False
args.seq_path = os.path.join("." + args.seq_path, "testingOnly")

# data directories
processed_dir = "../data/processed/"

# Generate files
energy_demand, adj_mat = loadEnergyData(processed_dir, incl_nodes = 1050, partial = False)
_, val_dataset = getDatasets(args, energy_demand, validation_range, validation_only = True)



