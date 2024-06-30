# 6.22 AlexNet
# 
# 初步尝试 使用在前向过程中：
# 1.输出精度
# 2.修改精度
# AlexNet铺平 至少把前向过程展开

import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
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
            nn.AdaptiveAvgPool2d((1, 1))   # 修改2  移除或调整原本的MaxPool2d，这里采用自适应池化
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
        
        # 定义卷积层和池化层
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
        
        # 定义全连接层
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
                
                
import torch.nn as nn
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
                
def alexnet():
    # return AlexNet()
    return AlexNet_flaten()

def alexnet_imagenet():
    return AlexNet_imagenet()