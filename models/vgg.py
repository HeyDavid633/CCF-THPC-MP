# 7.04 Vgg
# 只保留Vgg16的相关实现 
# 
# VGG16 原始定义 适应于CIFAR-100数据集
# VGG16_EMP 
# 

import torch
import torch.nn as nn
from utils import policy_string_analyze

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output
    
    
    def forward_with_print(self, x):
        print("\nMix Precision Info Output:")
        output = self.features(x)
        print("Output from features: \t", output.dtype)
        
        output = output.view(output.size()[0], -1)
        print("Output from view(): \t", output.dtype)
        
        output = self.classifier(output)
        print("Output from classi: \t", output.dtype)
        
        return output
    
def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG11, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512, 4096)
        # self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # 注意调整输入维度以匹配特征图尺寸
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool4(x)
        
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def forward_with_print(self, x):
        x = self.conv1(x)
        print("\nOutput from Conv1_1: ", x.dtype)
        x = self.bn1(x)
        print("Output from BatchNorm1: ", x.dtype)
        x = self.relu1(x)
        print("Output from ReLU1_1: ", x.dtype)
        x = self.pool1(x)
        print("Output from Pool1: ", x.dtype)
        
        x = self.conv2(x)
        print("Output from Conv2_1: ", x.dtype)
        x = self.bn2(x)
        print("Output from BatchNorm2: ", x.dtype)
        x = self.relu2(x)
        print("Output from ReLU2_1: ", x.dtype)
        x = self.pool2(x)
        print("Output from Pool2: ", x.dtype)
        
        x = self.conv3(x)
        print("Output from Conv3_1: ", x.dtype)
        x = self.bn3(x)
        print("Output from BatchNorm3: ", x.dtype)
        x = self.relu3(x)
        print("Output from ReLU3_1: ", x.dtype)
        x = self.conv4(x)
        print("Output from Conv3_2: ", x.dtype)
        x = self.bn4(x)
        print("Output from BatchNorm4: ", x.dtype)
        x = self.relu4(x)
        print("Output from ReLU3_2: ", x.dtype)
        x = self.pool3(x)
        print("Output from Pool3: ", x.dtype)
        
        x = self.conv5(x)
        print("Output from Conv14_1: ", x.dtype)
        x = self.bn5(x)
        print("Output from BatchNorm5: ", x.dtype)
        x = self.relu5(x)
        print("Output from ReLU4_1: ", x.dtype)
        x = self.conv6(x)
        print("Output from Conv4_2: ", x.dtype)
        x = self.bn6(x)
        print("Output from BatchNorm6: ", x.dtype)
        x = self.relu6(x)
        print("Output from ReLU4_2: ", x.dtype)
        x = self.pool4(x)
        print("Output from Pool4: ", x.dtype)
        
        x = self.conv7(x)
        print("Output from Conv5_1: ", x.dtype)
        x = self.bn7(x)
        print("Output from BatchNorm7: ", x.dtype)
        x = self.relu7(x)
        print("Output from ReLU5_1: ", x.dtype)
        x = self.conv7(x)
        print("Output from Conv5_2: ", x.dtype)
        x = self.bn8(x)
        print("Output from BatchNorm8: ", x.dtype)
        x = self.relu8(x)
        print("Output from ReLU5_2: ", x.dtype)
        x = self.pool5(x)
        print("Output from Pool5: ", x.dtype)

        x = x.view(x.size(0), -1)
        print("Output from View: ", x.dtype)
        
        x = self.fc1(x)
        print("Output from FC1: ", x.dtype)
        x = self.relu_fc1(x)
        print("Output from ReLU_FC1: ", x.dtype)
        x = self.dropout1(x)
        print("Output from Dropout1: ", x.dtype)
        
        x = self.fc2(x)
        print("Output from FC2: ", x.dtype)
        x = self.relu_fc2(x)
        print("Output from ReLU_FC2: ", x.dtype)
        x = self.dropout2(x)
        print("Output from Dropout2: ", x.dtype)
        
        x = self.fc3(x)
        print("Output from FC3: ", x.dtype)
            
        
        return x
        

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.fc1 = nn.Linear(512, 4096)
        # self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # 注意调整输入维度以匹配特征图尺寸
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)
        
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)
        
        x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def forward_with_print(self, x):
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        print("Output from relu1_1: ", x.dtype)
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        print("Output from relu1_2: ", x.dtype)
        x = self.pool1(x)
        print("Output from pool1: ", x.dtype)
        
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        print("Output from relu2_1: ", x.dtype)
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        print("Output from relu2_2: ", x.dtype)
        x = self.pool2(x)
        print("Output from pool2: ", x.dtype)
        
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        print("Output from relu3_1: ", x.dtype)
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        print("Output from relu3_2: ", x.dtype)
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        print("Output from relu3_3: ", x.dtype)
        x = self.pool3(x)
        print("Output from pool3: ", x.dtype)
        
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        print("Output from relu4_1: ", x.dtype)
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        print("Output from relu4_2: ", x.dtype)
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        print("Output from relu4_3: ", x.dtype)
        x = self.pool4(x)
        print("Output from pool4: ", x.dtype)
        
        x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        print("Output from relu5_1: ", x.dtype)
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        print("Output from relu5_2: ", x.dtype)
        x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        print("Output from relu5_3: ", x.dtype)
        x = self.pool5(x)
        print("Output from pool5: ", x.dtype)

        x = x.view(x.size(0), -1)
        print("Output from View: ", x.dtype)
        
        x = self.fc1(x)
        print("Output from FC1: ", x.dtype)
        x = self.relu_fc1(x)
        print("Output from ReLU_FC1: ", x.dtype)
        x = self.dropout1(x)
        print("Output from Dropout1: ", x.dtype)
        
        x = self.fc2(x)
        print("Output from FC2: ", x.dtype)
        x = self.relu_fc2(x)
        print("Output from ReLU_FC2: ", x.dtype)
        x = self.dropout2(x)
        print("Output from Dropout2: ", x.dtype)
        
        x = self.fc3(x)
        print("Output from FC3: ", x.dtype)
            
        
        return x

class VGG16_EMP(nn.Module):
    def __init__(self, num_classes=1000, policy_precision_string = '00000000000000000000'+'00000000000000000000'+'00000000000'):
        super(VGG16_EMP, self).__init__()
    
        self.datatype_policy = policy_string_analyze(policy_precision_string)
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, dtype=self.datatype_policy[0])
        self.bn1_1 = nn.BatchNorm2d(64, dtype=self.datatype_policy[1])
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dtype=self.datatype_policy[3])
        self.bn1_2 = nn.BatchNorm2d(64, dtype=self.datatype_policy[4])
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, dtype=self.datatype_policy[7])
        self.bn2_1 = nn.BatchNorm2d(128, dtype=self.datatype_policy[8])
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, dtype=self.datatype_policy[10])
        self.bn2_2 = nn.BatchNorm2d(128, dtype=self.datatype_policy[11])
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[14])
        self.bn3_1 = nn.BatchNorm2d(256, dtype=self.datatype_policy[15])
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[17])
        self.bn3_2 = nn.BatchNorm2d(256, dtype=self.datatype_policy[18])
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, dtype=self.datatype_policy[20])
        self.bn3_3 = nn.BatchNorm2d(256, dtype=self.datatype_policy[21])
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[24])
        self.bn4_1 = nn.BatchNorm2d(512, dtype=self.datatype_policy[25])
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[27])
        self.bn4_2 = nn.BatchNorm2d(512, dtype=self.datatype_policy[28])
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[30])
        self.bn4_3 = nn.BatchNorm2d(512, dtype=self.datatype_policy[31])
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[34])
        self.bn5_1 = nn.BatchNorm2d(512, dtype=self.datatype_policy[35])
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[37])
        self.bn5_2 = nn.BatchNorm2d(512, dtype=self.datatype_policy[38])
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dtype=self.datatype_policy[40])
        self.bn5_3 = nn.BatchNorm2d(512, dtype=self.datatype_policy[41])
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.fc1 = nn.Linear(512, 4096, dtype=self.datatype_policy[44])
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096, dtype=self.datatype_policy[47])
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes, dtype=self.datatype_policy[50])
        
        
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
            
            if i == 43:
                x = x.view(x.size(0), -1)
                
        return x
    
    def forward_with_print(self, x):
        if self.datatype_policy[0] == torch.float16:     # 初始类型转换
            x = x.to(torch.float16)
        
        for i, module in enumerate(self.children()): 
            layer_name = module._get_name()
            if i == 0: print("\nEMP Precision Policy --------------------")
            print(f"{i+1:2d}\t{layer_name:10s}\t{x.dtype}")    # 输出每层精度信息
            
            if i < len(list(self.children())) - 1:
                if self.datatype_policy[i+1] != x.dtype:
                    if self.datatype_policy[i+1] == torch.float16:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)
            
            x = module(x)
            if i == 43:
                x = x.view(x.size(0), -1)
        return x


def vgg11_bn():
    # return VGG(make_layers(cfg['A'], batch_norm=True))
    return VGG11()

def vgg16_bn():
    # return VGG(make_layers(cfg['D'], batch_norm=True))
    return VGG16()

def vgg16_bn_emp(policy_precision_string):
    return VGG16_EMP(policy_precision_string = policy_precision_string)

