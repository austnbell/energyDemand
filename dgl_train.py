import argparse, time
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from tqdm import tqdm
import scipy.sparse as sp
from datetime import datetime
import math
%matplotlib inline

from GCN import GCN
from torch.utils.data import DataLoader
from dataUtils import loadEnergyData, processData, energyDataset
from modelUtils import saveCheckpoint, loadCheckpoint, plotPredVsTrue

processed_dir = "./data/"
validation_range = ["2014-10-01 00:00:00", "2014-12-31 23:00:00"]
validation_range = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in validation_range]



def DGL_Process(df):
    grouped = df.groupby('time')
    inputs = []
    targets = []
    for time, group in tqdm(grouped):
        group.node = group.node.astype('int64')
        group = group.sort_values('node')


        node_targets = group.load.values
        node_features = group.loc[group.time==time,['solar_ecmwf','wind_ecmwf','holiday','hour','dow','month','year',
                                          'season','country','voltage']].values

        node_features = torch.LongTensor(node_features).unsqueeze(1)
        node_targets = torch.FloatTensor(node_targets)
           
        inputs.append(node_features)
        targets.append(node_targets)
    return torch.stack(inputs).transpose(0,1), \
            torch.stack(targets).transpose(0,1)


def getDatasets(energy_demand, validation_range):
    energy_demand['time'] = pd.to_datetime(energy_demand['time'], format='%Y-%m-%d %H:%M:%S')
    
    # extract validation and training sets
    train_df = energy_demand[energy_demand['time'] < validation_range[0]].reset_index(drop = True)
    val_df = energy_demand[(energy_demand['time'] >= validation_range[0]) & 
                           (energy_demand['time'] <= validation_range[1])].reset_index(drop = True)
    
    train_dataset = DGL_Process(train_df)
    valid_dataset = DGL_Process(val_df)
    train_dataset = train_dataset.type(torch.FloatTensor)
    valid_dataset = valid_dataset.type(torch.FloatTensor)

    return train_dataset,valid_dataset
        
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
     


def main(args):
    # load and preprocess dataset
    energy_demand, adj_mat = loadEnergyData(processed_dir, incl_nodes = 300, partial = True)
    energy_demand = processData(energy_demand)
    train_dataset,val_dataset = getDatasets(energy_demand,validation_range)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    


    adj_norm = normalizeAdjMat(adj_mat)
    g = nx.from_numpy_matrix(adj_norm.data.numpy())
    in_feats = 10
    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_predict =1,
                args.n_layers,
                F.relu,
                args.dropout)

    if cuda:
        model.cuda()
    criterion = torch.nn.MSELoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    train_loss = []
    val_loss = []
    val_best = 1
    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        for batch_idx, (features, target) in enumerate(train_loader):
            features = features.to(args['device'])
            target = target.to(args['device'])
        
            optimizer.zero_grad()
        
        
            predicted = model(features)
            loss = criterion(predicted, target)
            loss.backward()
            optimizer.step()
        
            # update tracking
            np_loss = loss.detach().cpu().numpy()
            avg_trn_loss.update(np_loss, args.batch_size)
            epoch_trn_loss.append(np_loss)
        
        val_predictions = []
        val_target = []
        with torch.no_grad():
            model.eval()
            for vbatch_idx, (vfeatures, vtarget) in enumerate(val_loader):
                vfeatures = vfeatures.to(args['device'])
                vtarget = vtarget.to(args['device'])
            
                vpreds = model(vfeatures)
                vloss = criterion(vpreds, vtarget)
            
            
                # storage and tracking
                np_vloss = vloss.detach().cpu().numpy()
                np_vpreds = vpreds.detach().cpu().numpy()
                np_vtarget = vtarget.detach().cpu().numpy()
                avg_val_loss.update(np_vloss, args.batch_size)
                epoch_val_loss.append(np_vloss)
                val_predictions.append(np_vpreds)
                val_target.append(np_vtarget)

        scheduler.step()
    
        # store epoch losses 
        train_loss.append(np.mean(epoch_trn_loss))
        val_loss.append(np.mean(epoch_val_loss))
        val_predictions = np.concatenate(val_predictions)
        val_target = np.concatenate(val_target)
    
        # TODO: If validation imporves then save model 
        if val_loss[-1] < val_best:
            val_best = val_loss[-1]
            saveCheckpoint(model, filename = "DGL_GCN.pth")

         # show results
        print("Current Training Loss: " + str(round(train_loss[-1], 8)))
        print("Current Validation Loss: " + str(round(val_loss[-1], 8)))
        print("\n")

    plt.title('Train & Validation MSE Loss')
    plt.plot(train_loss, label = "Training Loss")
    plt.plot(val_loss, label = "Validation Loss")
    plt.legend()
       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--device",type=str,default='cpu',
            help='device')
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--batch-size",type=int,default=64,
            help="batch size")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)