import torch
import torch.nn as nn
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)    
        
class PINNet(nn.Module):
    def __init__(self,input_size, pathway, num_fc):
        super(PINNet, self).__init__()
        
        num_pw = pathway.shape[0]
        self.pathway = pathway

        self.fc1 = nn.Sequential(nn.Linear(input_size, num_fc),
                                 nn.BatchNorm1d(num_fc),
                                 nn.LayerNorm(num_fc),
                                 nn.Tanh(),
                                 nn.Dropout(0.1))
        
        self.fc2 = nn.Sequential(nn.Linear(num_fc + num_pw, 64),
                                 nn.BatchNorm1d(64),
                                 nn.Tanh(),
                                 nn.Dropout(0.1))      
        
        self.fc3 = nn.Sequential(nn.Linear(64,2),
                                 nn.Softmax(dim = 1))    
        
        self.pw_linear = nn.Linear(input_size, num_pw)
        self.norm_gene = torch.sqrt(pathway.sum(axis=1))
        self.pw = nn.Sequential(nn.BatchNorm1d(num_pw),
                                nn.LayerNorm(num_pw),
                                nn.Tanh(),
                                nn.Dropout(0.1))
                                
        self.fc1.apply(weights_init)   
        self.fc2.apply(weights_init)        
        self.fc3.apply(weights_init) 
        self.pw_linear.apply(weights_init) 

    def forward(self, input):     
        x1 = self.fc1(input)
        self.pw_linear.weight = nn.Parameter(self.pw_linear.weight * self.pathway)
        x2 = self.pw_linear(input)
        x2 = x2/self.norm_gene
        x2 = self.pw(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.fc2(out)
        out = self.fc3(out)
        return out