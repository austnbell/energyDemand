# energyDemand
## COMSW4995: Austin Bell, Ziyin Wang, Malik Drabla 

./main.py - Implements STGCN for node level energy forcasting  
Running:   
```
python main.py 
```
Should suffice, as long as args are correctly defined under the given config file. If there are no train data, set the load_seq arg True.    
./configs - Configuration files for main.py args   
./DGL.ipynb - Implementation of GCN with DGL 
./data - Model training data  
./savedModels - Model checkpoints and training/testing statisitics
./evaluation - Evaluation scripts and statisitics
./dataPrep - One-off notebooks for dataset cleaning and merging

## Description:
GNNs and Benchmarks for Node-level Energy Load Forecasting.  
 * Implementation of STGNN variants using [pytorch](https://pytorch.org/)
 * Implementation of GCN using [DGL](dgl.ai)

## Dependencies:  

* random  
* pytorch  
* networkx  
* numpy  
* pandas  
* time    
* pickle  
* DGL  
* sklearn  

### DataSet:
* [RE-Europe](https://www.nature.com/articles/sdata2017175)
## TODO: 
- Explore incorporating seq2seq with STGCN (e.g., [Zhu et. al 2020](https://onlinelibrary.wiley.com/doi/abs/10.1002/ett.3890)) 
- Vectorize and include date metadata (holidays, weekend, season, average load, etc.) + solar and wind forecasts  
- Experiment tracking & hyperparam optimization 
