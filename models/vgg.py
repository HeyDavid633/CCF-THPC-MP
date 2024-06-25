"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

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
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
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
        x = self.relu1_1(self.conv1_1(x))
        x = self.pool1(x)
        
        x = self.relu2_1(self.conv2_1(x))
        x = self.pool2(x)
        
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.pool3(x)
        
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.pool4(x)
        
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def forward_with_print(self, x):
        x = self.conv1_1(x)
        print("\nOutput from Conv1_1: ", x.dtype)
        x = self.relu1_1(x)
        print("Output from ReLU1_1: ", x.dtype)
        x = self.pool1(x)
        print("Output from Pool1: ", x.dtype)
        
        x = self.conv2_1(x)
        print("Output from Conv2_1: ", x.dtype)
        x = self.relu2_1(x)
        print("Output from ReLU2_1: ", x.dtype)
        x = self.pool2(x)
        print("Output from Pool2: ", x.dtype)
        
        x = self.conv3_1(x)
        print("Output from Conv3_1: ", x.dtype)
        x = self.relu3_1(x)
        print("Output from ReLU3_1: ", x.dtype)
        x = self.conv3_2(x)
        print("Output from Conv3_2: ", x.dtype)
        x = self.relu3_2(x)
        print("Output from ReLU3_2: ", x.dtype)
        x = self.pool3(x)
        print("Output from Pool3: ", x.dtype)
        
        x = self.conv4_1(x)
        print("Output from Conv14_1: ", x.dtype)
        x = self.relu4_1(x)
        print("Output from ReLU4_1: ", x.dtype)
        x = self.conv4_2(x)
        print("Output from Conv4_2: ", x.dtype)
        x = self.relu4_2(x)
        print("Output from ReLU4_2: ", x.dtype)
        x = self.pool4(x)
        print("Output from Pool4: ", x.dtype)
        
        x = self.conv5_1(x)
        print("Output from Conv5_1: ", x.dtype)
        x = self.relu5_1(x)
        print("Output from ReLU5_1: ", x.dtype)
        x = self.conv5_2(x)
        print("Output from Conv5_2: ", x.dtype)
        x = self.relu5_2(x)
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
        


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))
    # return VGG11()

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


