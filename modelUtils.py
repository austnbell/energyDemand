# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:55:55 2020

@author: Austin Bell
"""
import os
import torch
import matplotlib.pyplot as plt

# Plot predictions and actual for visualization 
def plotPredVsTrue(target, preds, time_slot_idx, node):
    y_true = target[time_slot_idx][node]
    y_preds = preds[time_slot_idx][node]
    
    plt.title('Gold Standard Vs. Predictions')
    plt.plot(y_preds, label = "Prediction Load")
    plt.plot(y_true, label = "Gold Standard Load")
    plt.legend()
    plt.show()


# save and load
def saveCheckpoint(model, folder='savedModels/checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Saving Checkpoint...")
        torch.save({
            'state_dict' : model.state_dict(),
        }, filepath)

def loadCheckpoint(model, folder='savedModels/checkpoint', filename='checkpoint.pth.tar'):
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise("No model in path {}".format(filepath))
    map_location = None 
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


# stand in for argsparser
class dotDict(dict):
    def __getattr__(self, name):
        return self[name]