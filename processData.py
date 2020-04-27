# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:50:32 2020

functions related to processing the data 

@author: Austin Bell
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle

with open("../data/processed/node_stats.pkl", "rb") as f:
    node_stats = pickle.load(f)

# very basic preprocessing
def processData(df, sol_wind_type = "ecmwf"):
    label_encoder = preprocessing.LabelEncoder() 
    #scaler = MinMaxScaler()
    
    assert sol_wind_type in ['cosmo', 'ecmwf']

    # convert cat to numbers 
    df['season'] = label_encoder.fit_transform(df['season']) 
    #df['country'] = label_encoder.fit_transform(df['country'])
    df['node'] = df['node'].astype(int)
    
    # normalize all data except the node ids and time
    #non_normalized_cols = ["node", "time", "solar_ecmwf", "wind_ecmwf", "holiday"]
    #node_time = df[non_normalized_cols]
    #df = df.drop(columns = non_normalized_cols)
    
    #df_normalized = pd.DataFrame(preprocessing.normalize(df), columns = df.columns)
    #df_normalized = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    
    #df_out = pd.concat([node_time, df_normalized], axis = 1)
    return df

# process the metadata
# extracts metadata for day of week, season, and node
# extracts whether it is a holiday
def processMetaData(X, y):
    # takes both subsets and returns a vector
    
    # get summary stats 
    node = str(X.node.unique()[0])
    season = int(y.season.unique()[0])
    dow = int(y.dow.unique()[0])
    
    stats = node_stats[(node, season, dow)]
    node_season_dow_mean = stats['load_mean']
    node_season_dow_var = stats['load_var']
    
    # get holiday information 
    holiday = int(y.holiday.unique()[0])
    
    return np.array([node_season_dow_mean, node_season_dow_var, holiday])

# split our sequences to incorporate historical data and the predicted values 
def splitSequences(df, start_idx, subset_feats = None, historical = 72, forecast = 24):

    # extract all 0 and 12 hour indices
    df = df.drop(columns = "time")
    last_idx = df.index[-1]
    
    X_sequences = []
    target = []
    metadata = []
    """
    This is extremely inefficient right now, but prioritizing interpretability and easiness to code right now
    Will probably need to update once we have settle on our final format
    """
    for i, idx in enumerate(start_idx):
        X = df.iloc[idx : idx+historical,:]
        y = df.iloc[idx+historical : idx+historical+forecast-1, :]
        
        X_nodes = list(X.node.unique())
        y_nodes = list(y.node.unique())
        
        # extract metadata
        meta_vec = processMetaData(X, y)
        
        # only include a subset of featurres
        if subset_feats is not None:
            X = X.loc[:, subset_feats]

        # subset to relevant features for y 
        y = y.loc[:,["load", "node"]]

        if len(X_nodes) > 1 or y_nodes != X_nodes:
            continue
        
        if idx+historical+forecast > last_idx:
            break
        
        # append - for now we are ignoring nodes
        X_sequences.append(np.array(X.drop(columns = "node")).astype(float).tolist())
        metadata.append(meta_vec)
        target.append(y.load.tolist())
        
        
    return torch.from_numpy(np.array(X_sequences)), \
            torch.from_numpy(np.array(target)), \
            torch.from_numpy(np.array(metadata))
            
   
