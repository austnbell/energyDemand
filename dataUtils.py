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
from torch.utils.data import Dataset
import math, gc, os, re 
from processData import processData, splitSequences

# Load feature data and graph data
def loadEnergyData(processed_dir, incl_nodes = "All", partial = False):
    # load features
    if partial:
        energy_demand = pd.read_parquet(processed_dir + "EnergyDemandData_Partial.parquet")
    else:
        energy_demand = pd.read_parquet(processed_dir + "EnergyDemandData.parquet")
        
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
def getDatasets(args, energy_demand = None, validation_range = None, validation_only = False):
      
    # get dataset shells
    train_dataset = energyDataset(args,
                                  historical_len = args.historical_input, 
                                  forecast_len = args.forecast_output, 
                                  processing_function = args.processing_function,
                                  data_type = "train")
    
    
    val_dataset = energyDataset(args, 
                                historical_len = args.historical_input, 
                                forecast_len = args.forecast_output, 
                                processing_function = args.processing_function,
                                data_type = "val")
    
    # load or generate our sequences
    if args.load_seq:
        if not validation_only:
            train_dataset.loadSequences()
        val_dataset.loadSequences()
        
    else: 
        assert energy_demand is not None and validation_range is not None
        energy_demand['time'] = pd.to_datetime(energy_demand['time'], format='%Y-%m-%d %H:%M:%S')
    
        # extract validation and training sets
        train_df = energy_demand[energy_demand['time'] < validation_range[0]].reset_index(drop = True)
        val_df = energy_demand[(energy_demand['time'] >= validation_range[0]) & 
                               (energy_demand['time'] <= validation_range[1])].reset_index(drop = True)
        
      
        # generate our actual sequences
        if not validation_only:
            train_dataset.generateSequences(train_df)
        val_dataset.generateSequences(val_df)

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








# core dataset class - only set up for LSTM now
class energyDataset(Dataset):
    def __init__(self, args, start_values = [0],#[0, 12], 
                 historical_len = 72, forecast_len = 24, 
                 processing_function = None, data_type = None):
        
        # init
        self.save_seq = args.save_seq
        if self.save_seq:
            self.seq_path = args.seq_path
        
        self.data_type = data_type
        self.args = args
        self.historical_len = historical_len
        self.forecast_len = forecast_len
        
        
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.metadata[index], self.target[index]
    
    # load our sequences from file paths
    def loadSequences(self):
        files = os.listdir(self.args.seq_path)
        
        # subset to datatype and file type
        files = [f for f in files if re.search(self.data_type, f)]
        data_files = [f for f in files if re.search("_data", f)]
        meta_files = [f for f in files if re.search("metadata", f)]
        target_files = [f for f in files if re.search("target", f)]
        
        # now we sort to ensure nodes are in the correct order (i.e., numeric ascending - node 1, node 2, node 3, ...)
        data_files = [[file, int(re.search("\d{1,5}", file).group(0))] for file in data_files]
        data_files = sorted(data_files, key = lambda x: x[1])
        data_files = [file[0] for file in data_files]

        target_files = [[file, int(re.search("\d{1,5}", file).group(0))] for file in target_files]
        target_files = sorted(target_files, key = lambda x: x[1])
        target_files = [file[0] for file in target_files]
        
        meta_files = [[file, int(re.search("\d{1,5}", file).group(0))] for file in meta_files]
        meta_files = sorted(meta_files, key = lambda x: x[1])
        meta_files = [file[0] for file in meta_files]
        
        # now load and add to dataset
        inputs = []
        targets = []
        metadata = []
        for input_file, target_file, meta_file in zip(data_files, target_files, meta_files):
            inputs.append(torch.load(os.path.join(self.args.seq_path, input_file)))
            targets.append(torch.load(os.path.join(self.args.seq_path, target_file)))
            metadata.append(torch.load(os.path.join(self.args.seq_path, meta_file)))
            
        self.inputs = torch.stack(inputs).transpose(0,1).type(torch.FloatTensor)
        self.target = torch.stack(targets).transpose(0,1).type(torch.FloatTensor)
        self.metadata = torch.stack(metadata).transpose(0,1).type(torch.FloatTensor)
        
        if self.args.subset_feats is not None and len(self.args.subset_feats) == 2:
                self.inputs = self.inputs.squeeze()
    
    # create our sequences from our base dataset
    def generateSequences(self, df):
        # TODO: allowing passing of parameters to processign funtion if necessary
        if self.args.processing_function is not None:
            df = self.args.processing_function(df)
            print("Processed Data")
            
        # do not return anything if we are saving - conserve memory
        if self.save_seq:
            self.formatGNNInput(df, self.args)
        else:
            self.inputs, self.target, self.metadata = self.formatGNNInput(df, self.args)
            self.inputs = self.inputs.type(torch.FloatTensor)
            self.target = self.target.type(torch.FloatTensor)
            self.metadata = self.metadata.type(torch.FloatTensor)
            
            if self.args.subset_feats is not None and len(self.args.subset_feats) == 2:
                self.inputs = self.inputs.squeeze()
    
        print("Generated Sequences")
    
    def formatGNNInput(self, df, args):
        """
        splits our dataframe into separate nodes and then splits the sequences for each
        Converts 3d input (num_obs, time, variables) for each node into a 3d input (num_obs, node, time, variables)
        Final input into model is 4d (batch size, node, time, variables)
        """

        # split our processed dataframes
        split_dfs = [x for _, x in df.groupby('node')]
        del df
        gc.collect()
        
        # for each group we generate our time series sequences 
        inputs, targets, metadata = [],[],[]
        for d in split_dfs:
            d = d.reset_index()
            node = d.node[0]
            print(node)
            
            start_idx =  d.index[d['hour'].isin([0,12])].tolist()
            
            inputs_tmp, target_tmp, meta_tmp = splitSequences(d, start_idx, 
                                                        subset_feats = args.subset_feats,
                                                        historical = self.historical_len,
                                                        forecast = self.forecast_len)       
            
            
            if self.save_seq:
                # save our node data
                torch.save(inputs_tmp, os.path.join(self.seq_path, 'node_seq_data_' + str(node) + "_" + str(self.data_type) + '.pt'))
                torch.save(meta_tmp, os.path.join(self.seq_path, 'node_seq_metadata_' + str(node) + "_" + str(self.data_type) + '.pt'))
                torch.save(target_tmp, os.path.join(self.seq_path, 'node_seq_target_' + str(node) + "_" + str(self.data_type) + '.pt'))
                
            
            else:
                inputs.append(inputs_tmp)
                targets.append(target_tmp)
                metadata.append(meta_temp)
                
        if not self.save_seq:
            return torch.stack(inputs).transpose(0,1), \
                    torch.stack(targets).transpose(0,1), \
                    torch.stack(metadata).transpose(0,1)
            
   