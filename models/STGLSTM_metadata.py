import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBlock(nn.Module):
    """
    Temporal Convolution --> LSTM
    """
    def __init__(self, in_feats, hidden_dim = 128, bidirectional = False):
        super(LSTMBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        ## input to LSTM is (batch,timestep_len, num_feats)
        self.lstm = nn.LSTM(in_feats, 
                            hidden_size = hidden_dim ,
                            num_layers = 1,
                            bidirectional = bidirectional,
                            #dropout = dropout,
                            batch_first=True)
        
        # attention layer
        self.attn_weights = nn.Parameter(torch.Tensor(1, hidden_dim),
                                     requires_grad=True)

        nn.init.xavier_uniform_(self.attn_weights.data)

        
    def forward(self, X, activation = True):
        # input into LSTM is (batch,timestep_len, num_feats)
        # we feed the network (batch_size, num_nodes, timestep_len, num_feats) - so we need to change
        # we need to do lstm for every node 

        X = X.split(1,dim=1)
        lstm_concat = []
        Hs = []
        for node_feats in X:
            lstm_out, hs = self.lstm(node_feats.squeeze(1))
            if self.bidirectional:
                lstm_out = (lstm_out[:, :, :self.hidden_dim] + lstm_out[:, :, self.hidden_dim:])/2 # averarge the two directions
            
            lstm_concat.append(lstm_out)
            Hs.append(hs)
            
        output = torch.stack(lstm_concat)

        return output.permute(1,0,2,3)

class spatioTemporalBlock(nn.Module):
    """
    Spatial Temporal Block to populate our STGCN
    """
    def __init__(self, num_nodes, in_feats, spatial_feats, args):
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
        self.t1 = LSTMBlock(in_feats, args.hidden_dim, args.bidirectional)
        
        # when testing on GCP, we will want to use pytorch geometric so that we can test out better GNNs
        # but pytorch geometric requires at least CUDA 9.2, but I only have 9.0
        # so for now, I am just copying someones GCN implementation based on 1st order approximation 
        # taken from creator of GCNs https://github.com/tkipf/pygcn
        #self.gconv = GCNConv(out_feats, spatial_feats)
        self.theta = nn.Parameter(torch.FloatTensor(int(args.hidden_dim) , spatial_feats))
        
        self.t2 = LSTMBlock(spatial_feats, args.hidden_dim, args.bidirectional)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    # reset spectral kernel
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)
           
    def forward(self, X, adj_norm):
        temp1 = self.t1(X)
        
        support = torch.einsum("ij,jklm->kilm", [adj_norm, temp1.permute(1, 0, 2, 3)])
        spatial1 = torch.matmul(support, self.theta) 
        
        out = self.t2(F.relu(spatial1))
        return self.batch_norm(out)



class STGNN(nn.Module):
    """
    bringing everything together
    """
    def __init__(self, num_nodes, in_feats, num_timesteps_in, 
                 num_timesteps_predict, args, Train = True):
        super(STGNN, self).__init__()
        
        self.dropout = nn.Dropout(int(args.dropout*Train))
        
        self.block1 = spatioTemporalBlock(num_nodes, 
                                          in_feats,
                                          args.spatial_feats,
                                          args)
        
        self.block2 = spatioTemporalBlock(num_nodes, 
                                          in_feats = int(args.hidden_dim ),
                                          spatial_feats = args.spatial_feats,
                                          args = args)
        
        # final temporal layor and output layer
        self.final_temporal = LSTMBlock(int(args.hidden_dim), 
                                        int(args.hidden_dim ), 
                                        args.bidirectional)
        
        self.fc_stglstm = nn.Linear(int(args.hidden_dim*num_timesteps_in) ,num_timesteps_predict) # no length loss for LSTM
        
        # metadata and output layer
        self.fc_metadata = nn.Linear(3, 1)
        self.fc_out = nn.Linear(num_timesteps_predict,num_timesteps_predict)
        
        
    def forward(self, features, metadata, adj_norm):
        features = self.dropout(features)
        h1 = self.block1(features, adj_norm)
        h2 = self.block2(h1, adj_norm)
        h3 = self.final_temporal(h2)
        
        stglstm_out = torch.sigmoid(
                self.fc_stglstm(h3.reshape((h3.shape[0], h3.shape[1], h3.shape[2] * h3.shape[3])))
                )
        
        # metadata
        h_meta = self.fc_metadata(metadata)
        out = self.fc_out(stglstm_out.add(h_meta))
        
        
        return out.squeeze()