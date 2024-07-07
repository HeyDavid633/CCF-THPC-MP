import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import policy_string_analyze


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
class GCN_flatten(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super(GCN_flatten, self).__init__()
        
        self.conv1 = dglnn.GraphConv(in_size, hid_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = dglnn.GraphConv(hid_size, out_size)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = self.relu1(h)
        h = self.dropout1(h)
        h = self.conv2(g, h)
        h = self.dropout2(h)

        return h
    
class GCN_flatten_emp(nn.Module):
    def __init__(self, in_size, hid_size, out_size, policy_precision_string = '00000'):
        super(GCN_flatten_emp, self).__init__()
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        self.conv1 = dglnn.GraphConv(in_size, hid_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = dglnn.GraphConv(hid_size, out_size)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, g, features):

        h = self.conv1(g, features)
       
        if self.datatype_policy[1] != h.dtype:
            if self.datatype_policy[1] == torch.float16:
                h = h.to(torch.float16)
            else:   
                h = h.to(torch.float32)    
        h = self.relu1(h)
        
        if self.datatype_policy[2] != h.dtype:
            if self.datatype_policy[2] == torch.float16:
                h = h.to(torch.float16)
            else:   
                h = h.to(torch.float32)  
        h = self.dropout1(h)
        
        if self.datatype_policy[3] != torch.float32:
            h = h.to(torch.float32)  
        h = self.conv2(g, h)
        
        if self.datatype_policy[4] != h.dtype:
            if self.datatype_policy[4] == torch.float16:
                h = h.to(torch.float16)
            else:   
                h = h.to(torch.float32)  
        h = self.dropout2(h)

        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            ))
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            ))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class GAT_flatten(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super(GAT_flatten, self).__init__()
        
        self.conv1 = dglnn.GATConv(in_size, hid_size, heads[0],
            feat_drop=0.6, attn_drop=0.6, activation=F.elu,)
        
        self.conv2 = dglnn.GATConv(hid_size * heads[0], out_size, heads[1],  
            feat_drop=0.6, attn_drop=0.6, activation=None,)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = h.flatten(1)
        h = self.conv2(g, h)
        h = h.mean(1)
        
        return h
    
class GAT_flatten_emp(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, policy_precision_string = '0000'):
        super(GAT_flatten_emp, self).__init__()
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        self.conv1 = dglnn.GATConv(in_size, hid_size, heads[0],
            feat_drop=0.6, attn_drop=0.6, activation=F.elu,)
        
        self.conv2 = dglnn.GATConv(hid_size * heads[0], out_size, heads[1],  
            feat_drop=0.6, attn_drop=0.6, activation=None,)

    def forward(self, g, inputs):
        
        h = self.conv1(g, inputs)
        
        if self.datatype_policy[1] != h.dtype:
            if self.datatype_policy[1] == torch.float16:
                h = h.to(torch.float16)
            else:   
                h = h.to(torch.float32)  
        h = h.flatten(1)
        
        if self.datatype_policy[3] != torch.float32:
            h = h.to(torch.float32)
        h = self.conv2(g, h)
        
        if self.datatype_policy[3] != h.dtype:
            if self.datatype_policy[3] == torch.float16:
                h = h.to(torch.float16)
            else:   
                h = h.to(torch.float32)  
        h = h.mean(1)
        
        return h