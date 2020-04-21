class LSTMBlock(nn.Module):
    """
    Temporal Convolution --> LSTM
    """
    def __init__(self, in_feats,hidden_dim = 128, n_layers = 1,bidirectional = False, dropout = 0):
        super(LSTMBlock, self).__init__()

        ## input to LSTM is (batch,timestep_len, num_feats)
        self.rnn = nn.LSTM(in_feats,hidden_size=hidden_dim,num_layers=n_layers,bidirectional=bidirectional,dropout=dropout,batch_first=True)

        
    def forward(self, X, activation = True):
        # input into LSTM is (batch,timestep_len, num_feats)
        # we feed the network (batch_size, num_nodes, timestep_len, num_feats) - so we need to change
        # we need to do lstm for every node 
        num_nodes = X.size(1)
        timestep_len = X.size(2)
        num_feats = X.size(3)
        X = X.split(1,dim=1)
        Output = []
        Hs = []
        for item in X:
            output, hs = self.rnn(item.squeeze(1))
            Output.append(output)
            Hs.append(hs)
        #(num_nodes,batch_size,timestep_len,hidden_size=128)
        Output = torch.stack(Output)
        #print(Output.size())
        #Output = Output.squeeze(4)


        
        return Output.permute(1,0,2,3)

class spatioTemporalBlock(nn.Module):
    """
    Spatial Temporal Block to populate our STGCN
    """
    def __init__(self, num_nodes, in_feats, hidden_dim,n_layers,bidirectional,dropout, spatial_feats, 
                bias = True):
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
        self.hidden_dim = hidden_dim
        self.t1 = LSTMBlock(in_feats, hidden_dim=hidden_dim,n_layers=n_layers,bidirectional=bidirectional,dropout=dropout)
        
        # when testing on GCP, we will want to use pytorch geometric so that we can test out better GNNs
        # but pytorch geometric requires at least CUDA 9.2, but I only have 9.0
        # so for now, I am just copying someones GCN implementation based on 1st order approximation 
        # taken from creator of GCNs https://github.com/tkipf/pygcn
        #self.gconv = GCNConv(out_feats, spatial_feats)
        self.theta = nn.Parameter(torch.FloatTensor(self.hidden_dim, spatial_feats))
        
        self.t2 = LSTMBlock(spatial_feats, hidden_dim=self.hidden_dim,n_layers=n_layers,bidirectional=bidirectional,dropout=dropout)
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
        #print(support.shape)
        #print(self.theta.shape)
        spatial1 = torch.matmul(support, self.theta) 
        #print("Spatial Block",  spatial1[0,0,:,:], spatial1[0,1,:,:])
        
        out = self.t2(F.relu(spatial1))
        #print("2nd Temporal Convolution", out[0,0,:,:], out[0,1,:,:])
        return self.batch_norm(out)



class STG2Seq(nn.Module):
    """
    bringing everything together
    """
    def __init__(self, num_nodes, in_feats, num_timesteps_in, 
                 num_timesteps_predict, hidden_dim,n_layers,bidirectional,dropout, spatial_feats, 
                bias = True):
        super(STG2Seq, self).__init__()
        
        self.block1 = spatioTemporalBlock(num_nodes, in_feats,
                                          hidden_dim = hidden_dim,
                                          n_layers = n_layers,
                                          bidirectional = bidirectional,
                                          dropout = dropout, 
                                          spatial_feats = 16,
                                          )
        
        self.block2 = spatioTemporalBlock(num_nodes, 
                                          in_feats = hidden_dim ,
                                          hidden_dim = hidden_dim,
                                          n_layers = n_layers,
                                          bidirectional = bidirectional,
                                          dropout = dropout, 
                                          spatial_feats = 16)
        
        # final temporal layor and output layer
        self.final_temporal = LSTMBlock(hidden_dim, 1,n_layers=n_layers,bidirectional=bidirectional,dropout=dropout)
        self.fc_out = nn.Linear(num_timesteps_in ,num_timesteps_predict) # no length loss for LSTM
        
    def forward(self, features, adj_norm):
        h1 = self.block1(features, adj_norm)
        h2 = self.block2(h1, adj_norm)
        #print(h2)
        h3 = self.final_temporal(h2)
        #print(h3[0,0,:,:],h3[0,1,:,:])
        #print(h3.size())
        out = self.fc_out(h3.reshape((h3.shape[0], h3.shape[1], -1,h3.shape[2])))
        #print(out[0,0,:],out[0,1,:])
        return out.squeeze()