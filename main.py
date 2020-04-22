# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:02:07 2020

currently does not work when we exceed 20 nodes 

@author: Austin Bell
"""


import os, sys
import argparse
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math, re
import gc
import importlib

# ML
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_classification.utils import Bar, AverageMeter


# user functions
from dataUtils import loadEnergyData, energyDataset, getDatasets, normalizeAdjMat
from processData import processData
from models.baseSTGCN import STGCN as STGNN
from modelUtils import saveCheckpoint, loadCheckpoint, plotPredVsTrue, dotDict

########################################################################################
# Parameters
########################################################################################
torch.manual_seed(0)
np.random.seed(0)

# last three months
validation_range = ["2014-10-01 00:00:00", "2014-12-31 23:00:00"]
validation_range = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in validation_range]

##### Load our args
config_file = "STGCN_metadata_config"
c = importlib.import_module("configs."+config_file)
args = c.args

print(args)

# data directories
processed_dir = "./data/processed/"


########################################################################################
# Data Prep
########################################################################################

# only load energy demand if we are not loading our data
if args.load_seq:
    # get number of nodes to include
    files = os.listdir(args.seq_path)
    incl_nodes = max([int(re.search("\d{1,5}", f).group(0)) for f in files if re.search("\d", f)])
    
    print("loading data")
    _, adj_mat = loadEnergyData(processed_dir, incl_nodes = incl_nodes, partial = False)
    energy_demand = None
else:
    energy_demand, adj_mat = loadEnergyData(processed_dir, incl_nodes = 4, partial = True)
    pass

# format for pytorch
train_dataset, val_dataset = getDatasets(args, energy_demand, validation_range)

# stop if we are just save sequences
if args.save_seq:
    sys.exit()

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
print("loaded Data Loaders")


# normalized adjacency matrix with self loop
adj_norm = normalizeAdjMat(adj_mat)

########################################################################################
# Network Definition
########################################################################################
num_nodes = train_dataset.target.shape[1]
num_features = train_dataset.inputs.shape[3]

#del train_dataset, val_dataset, adj_mat
gc.collect()


# Model init
Gnet = STGNN(num_nodes,
             num_features,
             args.historical_input,
             args.forecast_output-1).to(device=args.device)

# SGD and Loss
optimizer = torch.optim.Adam(Gnet.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steps, gamma=0.5)

criterion = nn.MSELoss()


train_loss = []
val_loss = []
val_best = 1
########################################################################################
# Training the Network
########################################################################################

for epoch in range(args.epochs):
    print("Epoch Number: " + str(epoch + 1))
    
    
    # tracking 
    avg_trn_loss = AverageMeter()
    avg_val_loss = AverageMeter()
    
    epoch_trn_loss = []
    epoch_val_loss = []
    
    #bar = Bar('Training Graph Net', max=int(len(train_dataset.inputs)/args.batch_size))
    end = time.time()
    
    adj_norm = adj_norm.to(args.device) 
    
    
    ###########################################################
    # Training the network
    ###########################################################
    Gnet.train()
    for batch_idx, (features, metadata, target) in enumerate(train_loader):
        features = features.to(args.device)
        metadata = metadata.to(args.device)
        target = target.to(args.device)
        
        optimizer.zero_grad()
        
        
        predicted = Gnet(features, metadata, adj_norm)
        loss = criterion(predicted, target)
        
        loss.backward()
        optimizer.step()
        
        # update tracking
        np_loss = loss.detach().cpu().numpy()
        avg_trn_loss.update(np_loss, args.batch_size)
        epoch_trn_loss.append(np_loss)
        
        # plot progress 
        if batch_idx % 8 == 0:
            pass
            #print(f"Batch index {batch_idx} Out of {len(train_loader)}")
            #print(f"Average Training Loss: {avg_trn_loss.avg}")
    
     
    
    ###########################################################
    # Network validation
    ###########################################################
    # making some poor memory choices, but easier to debug for now
    val_predictions = []
    val_target = []
    with torch.no_grad():
        Gnet.eval()
        for vbatch_idx, (vfeatures, vmetadata, vtarget) in enumerate(val_loader):
            vfeatures = vfeatures.to(args.device)
            vmetadata = vmetadata.to(args.device)
            vtarget = vtarget.to(args.device)
            
            vpreds = Gnet(vfeatures, vmetadata, adj_norm)
            vloss = criterion(vpreds, vtarget)
            
            # TODO: un-normalize the loss and convert to MAE for better interpretation
            
            # storage and tracking
            np_vloss = vloss.detach().cpu().numpy()
            np_vpreds = vpreds.detach().cpu().numpy()
            np_vtarget = vtarget.detach().cpu().numpy()
            avg_val_loss.update(np_vloss, args.batch_size)
            epoch_val_loss.append(np_vloss)
            val_predictions.append(np_vpreds)
            val_target.append(np_vtarget)
            
            if batch_idx % 8 == 0:
                pass
                #print(f"Batch index {vbatch_idx} Out of {len(val_loader)}")
                #print(f"Average Validation Loss: {avg_val_loss.avg}")
            
            
    scheduler.step()
    
    # store epoch losses 
    train_loss.append(np.mean(epoch_trn_loss))
    val_loss.append(np.mean(epoch_val_loss))
    val_predictions = np.concatenate(val_predictions)
    val_target = np.concatenate(val_target)
    
    # TODO: If validation imporves then save model 
    if val_loss[-1] < val_best:
        val_best = val_loss[-1]
        saveCheckpoint(Gnet, filename = args.model_name)

    # show results
    #print(val_target[0][0], val_predictions[0][0])
    #print(val_target[0][1], val_predictions[0][1])
    print("Current Training Loss: " + str(round(train_loss[-1], 8)))
    print("Current Validation Loss: " + str(round(val_loss[-1], 8)))
    print("\n")
    

    
plt.title('Train & Validation MSE Loss')
plt.plot(train_loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend()
plt.show()
#plt.ylim(0,.1)
 
plotPredVsTrue(val_target, val_predictions, 10, 2)

#val_predictions[0][0], val_predictions[0][1]

#Gnet = loadCheckpoint(Gnet, filename = "initial model.pth")

# maybe will want to include later
# =============================================================================
#         bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_train: {l_trn:.4f} | Loss_v: {:.3f}'.format(
#                     batch=batch_idx,
#                     size=int(len(train_loader)),
#                     total=bar.elapsed_td,
#                     eta=bar.eta_td,
#                     l_trn=avg_trn_loss.avg,
#                     l_val=0.000)
#         bar.next()
# =============================================================================
