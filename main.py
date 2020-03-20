# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:02:07 2020

currently does not work when we exceed 20 nodes 

@author: Austin Bell
"""

import os
import argparse
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

# ML
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_classification.utils import Bar, AverageMeter


# user functions
from dataUtils import loadEnergyData, processData, energyDataset
from models.baseSTGCN import STGCN
from modelUtils import saveCheckpoint, loadCheckpoint, plotPredVsTrue

########################################################################################
# Parameters
########################################################################################
torch.manual_seed(0)
np.random.seed(0)

historical_input = 48
forecast_output = 24

# last three months - will add test set later
validation_range = ["2014-10-01 00:00:00", "2014-12-31 23:00:00"]
validation_range = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in validation_range]

# will replace with argparse later
args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 200,
        "batch_size": 64,
        "lr": .001
}

# data directories
processed_dir = "./data/processed/"

########################################################################################
# Data Prep
########################################################################################
energy_demand, adj_mat = loadEnergyData(processed_dir, incl_nodes = 300, partial = False)


# will add option for test loader too later
def getDatasets(energy_demand, validation_range, subset_x = None):
    energy_demand['time'] = pd.to_datetime(energy_demand['time'], format='%Y-%m-%d %H:%M:%S')
    
    # extract validation and training sets
    train_df = energy_demand[energy_demand['time'] < validation_range[0]].reset_index(drop = True)
    val_df = energy_demand[(energy_demand['time'] >= validation_range[0]) & 
                           (energy_demand['time'] <= validation_range[1])].reset_index(drop = True)
    
    
    # get dataloaders
    train_dataset = energyDataset(train_df, subset_x = subset_x,
                                  historical_len = historical_input, 
                                  forecast_len = forecast_output, 
                                  processing_function = processData)
    
    
    val_dataset = energyDataset(val_df, subset_x = subset_x,
                                historical_len = historical_input, 
                                forecast_len = forecast_output, 
                                processing_function = processData)
    

    return train_dataset, val_dataset

train_dataset, val_dataset = getDatasets(energy_demand, validation_range)

train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

# Normalized adjacency matrix with self loop 
# if A is not normalized we will completely change the scale of the feature vectors during our net
# since we do A*H*W
def normalizeAdjMat(adj_mat):
    # add self loop - ensures that a node's own features are included in calculations by creating an edge to itself
    n = adj_mat.shape[0]
    adj_mat = adj_mat +  np.diag(np.ones(n, dtype=np.float32))
    
    # generate node degree matrix
    D = np.zeros((n, n), float)
    np.fill_diagonal(D, np.sum(adj_mat, axis = 1))
    
    # get D^-(1/2)
    D_norm = D**(-(1/2))
    D_norm[D_norm==math.inf] =0  # handle infs 
    
    # Normalization formula is  D^(−1/2) * A * D^(−1/2)
    norm_adj_mat = np.matmul(np.matmul(D_norm, adj_mat), D_norm)  
    return torch.FloatTensor(norm_adj_mat)

adj_norm = normalizeAdjMat(adj_mat)

    


########################################################################################
# Network Definition
########################################################################################
num_nodes = len(energy_demand.node.unique())
num_features = train_dataset.inputs.shape[3]

# Model init
Gnet = STGCN(num_nodes,
             num_features,
             historical_input,
             forecast_output).to(device=args['device'])

# SGD and Loss
optimizer = torch.optim.Adam(Gnet.parameters(), lr=args['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.25)

criterion = nn.MSELoss()


train_loss = []
val_loss = []
val_best = 1
########################################################################################
# Training the Network
########################################################################################

for epoch in range(args['epochs']):
    print("Epoch Number: " + str(epoch + 1))
    
    
    # tracking 
    avg_trn_loss = AverageMeter()
    avg_val_loss = AverageMeter()
    
    epoch_trn_loss = []
    epoch_val_loss = []
    
    bar = Bar('Training Graph Net', max=int(len(train_dataset.inputs)/args['batch_size']))
    end = time.time()
    
    adj_norm = adj_norm.to(args['device']) 
    
    
    ###########################################################
    # Training the network
    ###########################################################
    Gnet.train()
    for batch_idx, (features, target) in enumerate(train_loader):
        features = features.to(args['device'])
        target = target.to(args['device'])
        
        optimizer.zero_grad()
        
        
        predicted = Gnet(features, adj_norm)
        loss = criterion(predicted, target)
        
        loss.backward()
        optimizer.step()
        
        # update tracking
        np_loss = loss.detach().cpu().numpy()
        avg_trn_loss.update(np_loss, args['batch_size'])
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
        for vbatch_idx, (vfeatures, vtarget) in enumerate(val_loader):
            vfeatures = vfeatures.to(args['device'])
            vtarget = vtarget.to(args['device'])
            
            vpreds = Gnet(vfeatures, adj_norm)
            vloss = criterion(vpreds, vtarget)
            
            # TODO: un-normalize the loss and convert to MAE for better interpretation
            
            # storage and tracking
            np_vloss = vloss.detach().cpu().numpy()
            np_vpreds = vpreds.detach().cpu().numpy()
            np_vtarget = vtarget.detach().cpu().numpy()
            avg_val_loss.update(np_vloss, args['batch_size'])
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
        saveCheckpoint(Gnet, filename = "initial model.pth")

    # show results
    print("Current Training Loss: " + str(round(train_loss[-1], 8)))
    print("Current Validation Loss: " + str(round(val_loss[-1], 8)))
    print("\n")
    

    
plt.title('Train & Validation MSE Loss')
plt.plot(train_loss, label = "Training Loss")
plt.plot(val_loss, label = "Validation Loss")
plt.legend()
#plt.ylim(0,.1)
 
#plotPredVsTrue(val_target, val_predictions, 18, 3)


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