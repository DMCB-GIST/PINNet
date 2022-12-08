import os, torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocessing (path, tissue):
    ### pathway datasets
    if (path == "GO"):
        pathway = pd.read_csv("../data/pathway_go_bp.csv", header=0)
    elif (path == "KEGG"):
        pathway = pd.read_csv("../data/pathway_kegg.csv", header=0)       
    print(">> Pathway Data :",path)

    pathway_info = pathway.iloc[:,1:]
    pathway_info = pathway_info.values
    pathway_info = np.transpose (pathway_info)
    pathway_info = torch.FloatTensor(pathway_info)
    print("pathway matrix shape : ",pathway_info.shape)
    print("num_pathway : ",pathway_info.shape[0])

    ### expression datasets
    print(">> Expression Data :",tissue)
    if (tissue == "brain"):
        data = pd.read_csv("../data/brain_expression.csv", header=0)

    expression = data.iloc[:,1:]
    gene = data.iloc[:,1]
    expression = expression.values
    expression = np.transpose(expression)

    scaler = MinMaxScaler()
    scaler = scaler.fit(expression)
    expression = scaler.transform(expression)

    sample_dim = expression.shape[0]
    input_dim = expression.shape[1]

    #print dimension of sample and number of genes
    print("sample_dim : ",sample_dim)
    print("input_size (number of genes): ",input_dim)
    
    if (tissue == "brain"):
        status = np.append(np.zeros((157)),np.ones((310)),axis = 0)
        status = status.reshape(467,1)

    patient = list(data.iloc[:,1:].columns.values.tolist()) 
    print("patient list : ",patient[1:6])
    
    return pathway_info, expression, status


pathway = "KEGG"
tissue = "brain"
pathway_info, expression, status = preprocessing(pathway, tissue)

trainArgs = {}
trainArgs['x_data'] = expression
trainArgs['y_data'] = status
trainArgs['pathway_info'] = pathway_info
# trainArgs['num_fc_list'] = [32, 64, 128]
# trainArgs['lr_list'] = [0.0001,0.0005,0.001]
trainArgs['num_fc_list'] = [32]
trainArgs['lr_list'] = [0.0001]
trainArgs['device'] = '0'
trainArgs['seed'] = 0
trainArgs['pathway'] = pathway
trainArgs['tissue'] = tissue
trainArgs['filename'] = 'result.csv'
