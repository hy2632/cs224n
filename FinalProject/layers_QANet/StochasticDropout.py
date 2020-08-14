import torch
import torch.nn as nn
import torch.nn.functional as F

class Stochastic_Dropout(nn.Module): 

    def __init__(self, final_survival_prob, depth):
        """Stochastic layer dropout 
            https://arxiv.org/pdf/1603.09382.pdf
            Args: 
                Final_survival_prob: survival probability of the last layer, p_L (in paper)
                depth: Depth of the stack of layer (number of layers)
                dropout_rate: regular drop out rate
        """ 
        super(Stochastic_Dropout, self).__init__()
        ######################## PARAMETERS #############################
        self.final_survival = final_survival_prob  
        self.depth = depth
    
    def forward(self, x, layer, norm, current_depth, multihead = (False, None)):
        """ NOTE: ALL ENCODER BLOCK RELUs ARE PERFORMED HERE
            Args: 
                @param x: skip connection
                @param layer: layer to be implemented
                @param norm: norm layer
                @param current_depth: depth at the current time
                @param multihead: tuple(bool indicating whether "layer" is multihead, mask required for multihead)
        """ 
        if self.training:
            survived = 1 - current_depth/self.depth*(1-self.final_survival) > torch.empty(1,1).uniform_(0,1)
            if survived and multihead[0]:
                temp = norm(x)
                temp = temp.permute(1, 0, 2)
                temp, _ = layer(temp, temp, temp, key_padding_mask= (multihead[1] == False))
                temp = temp.permute(1, 0, 2)
                return x + temp
            elif survived and not multihead[0]: 
                return x + F.relu(layer(norm(x)))
            elif not survived: 
                return x
        else:
            if multihead[0]:
                temp = norm(x)
                temp = temp.permute(1, 0, 2)
                temp, _ = layer(temp, temp, temp, key_padding_mask= (multihead[1] == False))
                temp = temp.permute(1, 0, 2)
                return x + temp
            else:
                return x + F.relu(layer(norm(x)))