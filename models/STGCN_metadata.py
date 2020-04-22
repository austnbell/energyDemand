# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:04:45 2020

Uses just historical load data to predict future load 


@author: Austin Bell
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import GCNConv

class temporalConv(nn.Module):
    """
    Temporal Convolution Block
    """
    def __init__(self, in_feats, out_feats, kernel_size = 3):
        super(temporalConv, self).__init__()

        """
        Right now, we are just doing one temporal convolution
        If we choose to stack these then we need to add a residual connection
        This involves adding left padding too, in order to maintain seq len
        """
        
        self.conv1 = nn.Conv2d(in_feats, out_feats, (1, kernel_size))  # causal with (1,k) kernel
        self.conv2 = nn.Conv2d(in_feats, out_feats, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_feats, out_feats, (1, kernel_size))

        
    def forward(self, X, activation = True):
        # input into conv2d is (batch_size, num_feats, num_nodes, timestep_len)
        # we feed the network (batch_size, num_nodes, timestep_len, num_feats) - so we need to change
        X = X.permute(0, 3, 1, 2)
        h = self.conv1(X) * torch.tanh(self.conv2(X))
        #h = F.relu(h1 + self.conv3(X))
        if activation:
            h = torch.tanh(h + self.conv3(X))
        
        return h.permute(0,2,3,1)


class spatioTemporalBlock(nn.Module):
    """
    Spatial Temporal Block to populate our STGCN
    """
    def __init__(self, num_nodes, in_feats, out_feats, spatial_feats, 
                 kernel_size = 3):
        super(spatioTemporalBlock, self).__init__()
        
        """
        definition:
            t2 ( RELU(spectral_kernel(t1(input))))
            batch norm the output
            
        Spectral graph convolution:
            extract eigenvectors from normalized laplacian
            extract eigenvectors from our spectral kernel
            multiply these two together
            
        I am just using pytorch geometric for this
        """
        
        self.t1 = temporalConv(in_feats, out_feats, kernel_size)
        
        # when testing on GCP, we will want to use pytorch geometric so that we can test out better GNNs
        # but pytorch geometric requires at least CUDA 9.2, but I only have 9.0
        # so for now, I am just copying someones GCN implementation based on 1st order approximation 
        # taken from creator of GCNs https://github.com/tkipf/pygcn
        #self.gconv = GCNConv(out_feats, spatial_feats)
        self.theta = nn.Parameter(torch.FloatTensor(out_feats, spatial_feats))
        
        self.t2 = temporalConv(spatial_feats, out_feats, kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    # reset spectral kernel
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)
           
    def forward(self, X, adj_norm):
        temp1 = self.t1(X)
        #print("temporal1 convolution:", temp1[0,0,:,:], temp1[0,1,:,:])
        
        #spatial1 = self.gconv(temp1, adj_norm)
        support = torch.einsum("ij,jklm->kilm", [adj_norm, temp1.permute(1, 0, 2, 3)])
        spatial1 = torch.matmul(support, self.theta) 
        #print("Spatial Block",  spatial1[0,0,:,:], spatial1[0,1,:,:])
        
        out = self.t2(F.relu(spatial1), activation = False)
        #print("2nd Temporal Convolution", out[0,0,:,:], out[0,1,:,:])
        return self.batch_norm(out)



class STGNN(nn.Module):
    """
    bringing everything together
    """
    def __init__(self, num_nodes, in_feats, num_timesteps_in, 
                 num_timesteps_predict,  args, kernel_size = 3):
        super(STGNN, self).__init__()
        
        self.block1 = spatioTemporalBlock(num_nodes, in_feats,
                                          out_feats = args.out_feats, 
                                          spatial_feats = args.spatial_feats,
                                          kernel_size = kernel_size)
        
        self.block2 = spatioTemporalBlock(num_nodes, 
                                          in_feats = args.out_feats ,
                                          out_feats = args.out_feats, 
                                          spatial_feats = args.spatial_feats,
                                          kernel_size= kernel_size)
        
        # final temporal layor and output layer
        self.final_temporal = temporalConv(64, 64, kernel_size)
        self.fc_stgcn = nn.Linear((num_timesteps_in - ((kernel_size - 1) * 5))*64 ,num_timesteps_predict) # accounts for the length lost every temporal conv
        
        # metadata and output layer
        self.fc_metadata = nn.Linear(3, 1)
        self.fc_out = nn.Linear(num_timesteps_predict,num_timesteps_predict)
        
        
    def forward(self, features, metadata, adj_norm):
        # STGCN
        h1 = self.block1(features, adj_norm)
        h2 = self.block2(h1, adj_norm)
        h3 = self.final_temporal(h2, activation = False)
        stgcn_out = torch.sigmoid(self.fc_stgcn(h3.reshape((h3.shape[0], h3.shape[1], -1))))
        
        # metadata
        h_meta = self.fc_metadata(metadata)
        out = self.fc_out(stgcn_out.add(h_meta))

        return out
