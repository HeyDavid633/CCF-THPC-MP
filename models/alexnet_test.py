import torch
import torch.nn as nn
from utils import policy_string_analyze

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, policy_precision_string='00000000000000000000'):
        super(AlexNet, self).__init__()
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dtype=self.datatype_policy[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1, dtype=self.datatype_policy[3])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, dtype=self.datatype_policy[6])
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[8])
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[10])
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        # 定义全连接层
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256, 4096, dtype=self.datatype_policy[14])
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096, dtype=self.datatype_policy[17])
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes, dtype=self.datatype_policy[19])
        
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
            
            if i == 13:
                x = x.view(x.size(0), -1)
            
        return x
    
    def forward_with_print(self, x):
        if self.datatype_policy[0] == torch.float16:     # 初始类型转换
            x = x.to(torch.float16)
        
        for i, module in enumerate(self.children()): 
            layer_name = module._get_name()
            if i == 0: print("EMP Precision Policy --------------------\n")
            print(f"{i+1:2d}\t{layer_name:10s}\t{x.dtype}")    # 输出每层精度信息
            
            if i < len(list(self.children())) - 1:
                if self.datatype_policy[i+1] != x.dtype:
                    if self.datatype_policy[i+1] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)
            
            x = module(x)
            if i == 13:
                x = x.view(x.size(0), -1)
            
        return x

def alexnet(policy_precision_string):
    return AlexNet(policy_precision_string = policy_precision_string)
    # return AlexNet_flaten()