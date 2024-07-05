# 7.03 AlexNet
# 
# AlexNet 原始定义(修改了一定的维度以适应CIFAR-100的数据集) --- 用来做对比测试 fp32 / AMP
# AlexNet_flaten 展开的AlexNet 
# AlexNet_EMP 可以给到输入串的AlexNet
#
# AlexNet_imagenet 原始定义 适应于ImageNet数据集
# AlexNet_EMP_imagenet EMP寻语

import torch
import torch.nn as nn
from utils import policy_string_analyze
#最原始的定义
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),  # 正确调整了第一个全连接层的输入维度
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)        
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = x.view(x.size(0), -1)  # 自适应池化后，直接展平
        x = self.classifier(x)
        return x
    
    def forward_with_print(self, x):
        x = self.features(x)
        print("\nOutput from features: ", x.dtype)
        
        # x = x.view(x.size(0), 256 * 6 * 6)
        x = x.view(x.size(0), -1)  # 自适应池化后，直接展平
        print("Output from view(): ", x.dtype)
        
        x = self.classifier(x)
        print("Output from classifier: ", x.dtype)
        return x
    
# 修改了 Conv2d 等的kernel_size与stride 以适应于CIFAR-100的 32x32图片尺寸大小
# 修改后成功收敛
class AlexNet_flaten(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_flaten, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        
        return x
    
    def forward_with_print(self, x):
        # 卷积层1
        x = self.conv1(x)
        print("\nOutput from Conv1: ", x.dtype)
        x = self.relu1(x)
        print("Output from ReLU1: ", x.dtype)
        x = self.pool1(x)
        print("Output from Pool1: ", x.dtype)
        # 卷积层2
        x = self.conv2(x)
        print("Output from Conv2: ", x.dtype)
        x = self.relu2(x)
        print("Output from ReLU2: ", x.dtype)
        x = self.pool2(x)
        print("Output from Pool2: ", x.dtype)
        # 卷积层3
        x = self.conv3(x)
        print("Output from Conv3: ", x.dtype)
        x = self.relu3(x)
        print("Output from ReLU3: ", x.dtype)
        # 卷积层4
        x = self.conv4(x)
        print("Output from Conv4: ", x.dtype)
        x = self.relu4(x)
        print("Output from ReLU4: ", x.dtype)
        # 卷积层5
        x = self.conv5(x)
        print("Output from Conv5: ", x.dtype)
        x = self.relu5(x)
        print("Output from ReLU5: ", x.dtype)
        x = self.pool3(x)
        print("Output from AdaptivePool: ", x.dtype)
        
        x = x.view(x.size(0), -1)
        print("Output from View: ", x.dtype)
        
        # 全连接层1之前的Dropout
        x = self.dropout1(x)
        print("Output from Dropout1: ", x.dtype)
        x = self.fc1(x)
        print("Output from FC1: ", x.dtype)
        x = self.relu_fc1(x)
        print("Output from ReLUFc1: ", x.dtype)
        # 全连接层2之前的Dropout
        x = self.dropout2(x)
        print("Output from Dropout2: ", x.dtype)
        x = self.fc2(x)
        print("Output from FC2: ", x.dtype)
        x = self.relu_fc2(x)
        print("Output from ReLUFc2: ", x.dtype)
        x = self.fc3(x)
        print("Output from FC3: ", x.dtype)
        
        return x
                
class AlexNet_EMP(nn.Module):
    def __init__(self, num_classes=1000, policy_precision_string='00000000000000000000'):
        super(AlexNet_EMP, self).__init__()
        
        self.stage1_op_name = ["Conv2d", "BatchNorm2d", "Linear"]
        self.stage1_op = []
        self.all_layer_op = []
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
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
            if i > 0 and i < len(list(self.children())) :
                if self.datatype_policy[i-1] != self.datatype_policy[i]:
                    if self.datatype_policy[i] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)
            x = module(x)
            
            if i == 13:
                x = x.view(x.size(0), -1)
            
        return x
    
    def forward_with_print(self, x):
        if self.datatype_policy[0] == torch.float16:
            x = x.to(torch.float16)

        for i, module in enumerate(self.children()):
            if i > 0 and i < len(list(self.children())) :
                if self.datatype_policy[i-1] != self.datatype_policy[i]:
                    if self.datatype_policy[i] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)
            x = module(x)
            
            if i == 13:
                x = x.view(x.size(0), -1)
                
            layer_name = module._get_name()
            # if i == 0: print("\nEMP Precision Policy "+"-"*50)
            # print(f"{i+1:2d}\t{layer_name:10s}\t{x.dtype}")

            if layer_name in self.stage1_op_name:
                self.stage1_op.append({"id": i, "layer_name": layer_name, "data_type": str(x.dtype)})
            self.all_layer_op.append({"id": i, "layer_name": layer_name, "data_type": str(x.dtype)})

        return self.stage1_op, self.all_layer_op, x
                
           
                                
class AlexNet_imagenet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_imagenet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), # 原来的定义
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)        
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
              
class AlexNet_EMP_imagenet(nn.Module):
    def __init__(self, num_classes=1000, policy_precision_string='00000000000000000000'):
        super(AlexNet_EMP_imagenet, self).__init__()
        
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, dtype=self.datatype_policy[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2, dtype=self.datatype_policy[3])
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, dtype=self.datatype_policy[6])
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[8])
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[10])
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 定义全连接层
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096, dtype=self.datatype_policy[14])
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
                x = x.view(x.size(0), 256 * 6 * 6)
            
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
                x = x.view(x.size(0), 256 * 6 * 6)
            
        return x
    
                
def alexnet():
    return AlexNet_flaten()

def alexnet_imagenet():
    return AlexNet_imagenet()

def alexnet_emp(policy_precision_string):
    return AlexNet_EMP(policy_precision_string = policy_precision_string)  

def alexnet_emp_imagenet(policy_precision_string):
    return AlexNet_EMP_imagenet(policy_precision_string = policy_precision_string)  


