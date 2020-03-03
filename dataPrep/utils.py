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
from torch.utils.data import Dataset, DataLoader


# very basic preprocessing
def processData(df, sol_wind_type = "ecmwf"):
    label_encoder = preprocessing.LabelEncoder() 
    
    assert sol_wind_type in ['cosmo', 'ecmwf']
    if sol_wind_type == 'cosmo':
        df = df.drop(columns = ['solar_ecmwf', 'wind_ecmwf'])
    else: 
        df = df.drop(columns = ['solar_cosmo', 'wind_cosmo'])
    
    # convert cat to numbers 
    df['season'] = label_encoder.fit_transform(df['season']) 
    df['country'] = label_encoder.fit_transform(df['country'])
    
    # normalize all data except the node ids and time
    non_normalized_cols = ["node", "time", "solar_ecmwf", "wind_ecmwf", "holiday"]
    node_time = df[non_normalized_cols]
    df = df.drop(columns = non_normalized_cols)
    df_normalized = pd.DataFrame(preprocessing.normalize(df), columns = df.columns)
    
    df = pd.concat([node_time, df_normalized], axis = 1)
    return df




# split our sequences to incorporate historical data and the predicted values 
def split_sequences(df, start_idx , historical = 72, forecast = 24):

    # extract all 0 and 12 hour indices
    df = df.drop(columns = "time")
    last_idx = df.index[-1]
    
    X_sequences = []
    target = []
    for i, idx in enumerate(start_idx):
        X = df.iloc[idx : idx+historical,:]
        y = df.loc[idx+historical : idx+historical+forecast, ["load", "node"]]
        
        X_nodes = list(X.node.unique())
        y_nodes = list(y.node.unique())

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
    def __init__(self, df, start_values = [0, 12], historical_len = 72,
                  forecast_len = 24, processing_function = None):
        
        self.start_idx =  df.index[df['hour'].isin([0,12])].tolist()
        
        # TODO: allowing passing of parameters to processign funtion if necessary
        if processing_function is not None:
            df = processing_function(df)
            print("Processed Data")
        
        self.historical_len = historical_len
        self.forecast_len = forecast_len
        self.inputs, self.target = split_sequences(df, self.start_idx, 
                                                   historical = historical_len,
                                                   forecast = forecast_len)
        print("Generated Sequences")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.target[index]
        
    
"""
processed_dir = "./data/processed/"

df = pd.read_parquet(processed_dir + "Energy Demand Data (Partial).parquet")
df.head()

processed = processData(df)
dataset = energyDataset(df, processing_function = processData)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


for x, y in dataloader:
    print(x.shape)
    print(y.shape)
    break

"""