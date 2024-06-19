import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
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
            
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 原来的定义
            # nn.MaxPool2d(kernel_size=2, stride=1)   # 修改1  匹配图像尺寸 CIFAR100:32x32 / AlexNet为ImageNet:224x224
            nn.AdaptiveAvgPool2d((1, 1))   # 修改2  移除或调整原本的MaxPool2d，这里采用自适应池化
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096), # 原来的定义
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
    
def alexnet():
    return AlexNet()