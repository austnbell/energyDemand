# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:08:59 2020

@author: Austin Bell
"""
import torch
import sys
sys.path.append("..")
from modelUtils import dotDict
from dataUtils import loadEnergyData, processData, energyDataset, getDatasets, normalizeAdjMat



args = dotDict({
        # data params
        "historical_input": 24, # timestep inputs
        "forecast_output": 24, # timstep outputs
        "subset_feats": ['load', 'node', "solar_ecmwf"], # subset features to include? None is include all
        "save_seq": False, # save our sequences instead of splitting
        "load_seq": True, # load our sequences
        "seq_path": "./data/processed/nodeSequences", # path to saved sequences
        "processing_function": processData, # data processing function to use
        
        # model params
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 200,
        "batch_size": 64,
        "lr": .001,
        "steps": 50, 
        "model_name": "baselineSTGCN.pth"
})