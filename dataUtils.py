# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:45:05 2020

Utility scripts for loading data in our energy demand project

create a split sequences function then this can either be called by itself 
or put into a dataset class, converted to tensors, and extracted in batches 

currently implemented just for LSTM without spatial component. next step is to enable GNNs


@author: Austin Bell
"""

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import math

# Load feature data and graph data
def loadEnergyData(processed_dir, incl_nodes = "All", partial = False):
    # load features
    if partial:
        energy_demand = pd.read_parquet(processed_dir + "Energy Demand Data (Partial).parquet")
    else:
        energy_demand = pd.read_parquet(processed_dir + "Energy Demand Data.parquet")
        
    # load adjacency matrix
    adj_mat = np.load(processed_dir + "adjacency_matrix.npy")
    
    # subset for testing
    if incl_nodes != "All":
        assert type(incl_nodes) == int and incl_nodes <= 1514
        energy_demand = energy_demand.loc[energy_demand['node'].astype(int) <= incl_nodes] 
        num_nodes = len(energy_demand.node.unique())
        adj_mat = adj_mat[:num_nodes, :num_nodes]
        
    # I may need to do some normalization for the adjacency matrix here
        
    return energy_demand, adj_mat
  
# convert data into datasets
# will add option for test loader too later
def getDatasets(energy_demand, validation_range, historical_input, forecast_output, subset_x = None):
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




# very basic preprocessing
def processData(df, sol_wind_type = "ecmwf"):
    label_encoder = preprocessing.LabelEncoder() 
    scaler = MinMaxScaler()
    
    assert sol_wind_type in ['cosmo', 'ecmwf']
    if sol_wind_type == 'cosmo':
        df = df.drop(columns = ['solar_ecmwf', 'wind_ecmwf'])
    else: 
        df = df.drop(columns = ['solar_cosmo', 'wind_cosmo'])
    
    # convert cat to numbers 
    df['season'] = label_encoder.fit_transform(df['season']) 
    df['country'] = label_encoder.fit_transform(df['country'])
    df['node'] = df['node'].astype(int)
    
    # normalize all data except the node ids and time
    non_normalized_cols = ["node", "time", "solar_ecmwf", "wind_ecmwf", "holiday"]
    node_time = df[non_normalized_cols]
    df = df.drop(columns = non_normalized_cols)
    
    #df_normalized = pd.DataFrame(preprocessing.normalize(df), columns = df.columns)
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    
    df_out = pd.concat([node_time, df_normalized], axis = 1)
    return df_out



# split our sequences to incorporate historical data and the predicted values 
def splitSequences(df, start_idx, subset_x = None, historical = 72, forecast = 24):

    # extract all 0 and 12 hour indices
    df = df.drop(columns = "time")
    last_idx = df.index[-1]
    
    X_sequences = []
    target = []
    """
    This is extremely inefficient right now, but prioritizing interpretability and easiness to code right now
    Will probably need to update once we have settle on our final format
    """
    for i, idx in enumerate(start_idx):
        X = df.iloc[idx : idx+historical,:]
        
        y = df.loc[idx+historical+1 : idx+historical+forecast, ["load", "node"]]
        
        X_nodes = list(X.node.unique())
        y_nodes = list(y.node.unique())
        
        # only include a subset of featurres
        if subset_x is not None:
            X = X.loc[:, subset_x]

        if len(X_nodes) > 1 or y_nodes != X_nodes:
            continue
        
        if idx+historical+forecast > last_idx:
            break
        
        # append - for now we are ignoring nodes
        X_sequences.append(np.array(X.drop(columns = "node")).astype(float).tolist())
        target.append(y.load.tolist())
        
        
    return torch.from_numpy(np.array(X_sequences)), \
            torch.from_numpy(np.array(target))
   

# core dataset class - only set up for LSTM now
class energyDataset(Dataset):
    def __init__(self, df, start_values = [0, 12], subset_x = None, 
                 historical_len = 72, forecast_len = 24, 
                 processing_function = None):
        
        
        # TODO: allowing passing of parameters to processign funtion if necessary
        if processing_function is not None:
            df = processing_function(df)
            print("Processed Data")
        
        self.historical_len = historical_len
        self.forecast_len = forecast_len
        self.inputs, self.target = self.formatGNNInput(df, subset_x)
        self.inputs = self.inputs.type(torch.FloatTensor)
        self.target = self.target.type(torch.FloatTensor)
        
        if subset_x is not None and len(subset_x) == 2:
            self.inputs = self.inputs.squeeze()
        
        print("Generated Sequences")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.target[index]
    
    
    def formatGNNInput(self, df, subset_x):
        """
        splits our dataframe into separate nodes and then splits the sequences for each
        Converts 3d input (num_obs, time, variables) for each node into a 3d input (num_obs, node, time, variables)
        Final input into model is 4d (batch size, node, time, variables)
        """

        # split our processed dataframes
        split_dfs = [x for _, x in df.groupby('node')]
        
        inputs, targets = [],[]
        for d in split_dfs:
            d = d.reset_index()
            start_idx =  d.index[d['hour'].isin([0,12])].tolist()
            
            inputs_tmp, target_tmp = splitSequences(d, start_idx, subset_x = subset_x,
                                                     historical = self.historical_len,
                                                     forecast = self.forecast_len)       
            inputs.append(inputs_tmp)
            targets.append(target_tmp)
            
        return torch.stack(inputs).transpose(0,1), \
                torch.stack(targets).transpose(0,1)
        
   