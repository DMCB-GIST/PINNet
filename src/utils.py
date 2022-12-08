import torch, os
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_min = 0
        self.delta = delta
        self.path = path

    def __call__(self, val_auc, model, epoch):

        score = val_auc
        if epoch > 0 :
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model)
            elif score > self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, val_auc, model):
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_min:.6f} --> {val_auc:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_auc_min = val_auc
                
class CustomDataset(Dataset): 
    def __init__(self,x,y):
        self.x_data = x
        self.y_data = y

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])
        return x, y