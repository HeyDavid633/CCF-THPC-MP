# 7.08 GAN 网络的相关定义 
# 
#
import torch
import torch.nn as nn
from utils import policy_string_analyze

class Generator(nn.Module):
    def __init__(self): 
        super(Generator, self).__init__() 
        
        self.fc1 = nn.Linear(100, 256)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 28*28)
        self.tanh= nn.Tanh()
        
    def forward(self, x):             
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        x = x.view(-1, 28, 28) 
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 512)
        self.lkyrelu_fc1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 256)
        self.lkyrelu_fc2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):    
        x = x.view(-1, 28*28)
        x = self.lkyrelu_fc1(self.fc1(x))
        x = self.lkyrelu_fc2(self.fc2(x))
        x = self.fc3(x)
        x = self.sig(x)
        return x
            

class Generator_emp(nn.Module):
    def __init__(self, policy_precision_string='000000'): 
        super(Generator_emp, self).__init__() 
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        self.fc1 = nn.Linear(100, 256, dtype=self.datatype_policy[0])
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 512, dtype=self.datatype_policy[2])
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 28*28, dtype=self.datatype_policy[4])
        self.tanh = nn.Tanh()
        
    def forward(self, x):             
        if self.datatype_policy[0] == torch.float16:
            x = x.to(torch.float16)
        
        for i, module in enumerate(self.children()):    
            if i < len(list(self.children())) - 1:
                if self.datatype_policy[i+1] != x.dtype:
                    if self.datatype_policy[i+1] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)                
            x = module(x)
        
        x = x.view(-1, 28, 28) 
        return x
    
class Discriminator_emp(nn.Module):
    def __init__(self, policy_precision_string='000000'):
        super(Discriminator_emp, self).__init__()
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        self.fc1 = nn.Linear(28*28, 512, dtype=self.datatype_policy[0])
        self.lkyrelu_fc1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256, dtype=self.datatype_policy[2])
        self.lkyrelu_fc2 = nn.LeakyReLU(inplace=True)
        self.fc3 = nn.Linear(256, 1, dtype=self.datatype_policy[4])
        self.sig = nn.Sigmoid()
        
    def forward(self, x):    
        if self.datatype_policy[0] == torch.float16:
            x = x.to(torch.float16)
            
        x = x.view(-1, 28*28)
        
        for i, module in enumerate(self.children()):    
            if i < len(list(self.children())) - 1:
                if self.datatype_policy[i+1] != x.dtype:
                    if self.datatype_policy[i+1] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)                
            x = module(x)
        return x