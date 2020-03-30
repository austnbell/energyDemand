# energyDemand
COMSW4995: Austin Bell, Ziyin Wang, Malik Drabla 

./main.py - Implements STGCN for node level energy forcasting.  
./configs - Configuration files for main.py args . 
Running: 
  python main.py   
Should suffice, as long as args are correctly defined under the given config file. If there are no train data, set the load_seq arg True.  
./DGL.ipynb - Implementation of GCN with DGL 
./data - Model training data  
./savedModels - checkpoints and training/testing statisitics  

Description:
GNNs and Benchmarks for Node-level Load Forecasting
Pytorch implementation of node level energy forcasting using STGNN variants.

TODO:
- Explore incorporating seq2seq with STGCN (e.g., Zhu et. al 2020(=)
- Vectorize and include date metadata (holidays, weekend, season, average load, etc.) + solar and wind forecasts 
- Experiment tracking & hyperparam optimization 

Dependencies:

random
pytorch
networkx
numpy
pandas
time
pickle
DGL
sklearn

