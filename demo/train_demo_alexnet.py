# 6.19 train_demo_alexNet
# 
# 原始的AlexNet 可以训练 --- 数据集是随机的 不追求收敛
# 这个例子作为 打开 的基准 --- 因为维度是可以确定的
# 
# python train_demo_alexNet.py False
# 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
from torchsummary import summary
from datetime import datetime
# from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import sys

log_dir='./tb_log'
current_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def torch_cuda_active():
    if torch.cuda.is_available():
        print('PyTorch version\t:', torch.__version__)
        print('CUDA version\t:', torch.version.cuda)
        print('GPU\t\t:', torch.cuda.get_device_name(), '\n')
        return torch.device('cuda')
    else:
        print('CUDA is not available!')
        return torch.device('cpu')


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
            nn.MaxPool2d(kernel_size=3, stride=2),  
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), 
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

#更为扁平的展开代码
class AlexNet_flaten(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 定义全连接层
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu3(self.conv3(x))
        
        x = self.relu4(self.conv4(x))
        
        x = self.relu5(self.conv5(x))
        x = self.pool3(x)
        
        # 调整形状以便进入全连接层
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        x = self.relu_fc1(self.fc1(x))
        
        x = self.dropout2(x)
        x = self.relu_fc2(self.fc2(x))
        
        x = self.fc3(x)
        
        return x

def train_loop(model, criterion, optimizer, scaler=None, steps=100, use_amp=False):
    model.train()
    start_time = time.time()
    
    trace_file_name = f"{current_time_str}"
    tb_handler = torch.profiler.tensorboard_trace_handler(log_dir, worker_name=trace_file_name)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=tb_handler,
                record_shapes=True,  
                profile_memory=True, 
                with_stack=True) as prof:  

        for step in range(steps):
            inputs = torch.randn(64, 3, 224, 224, device=device)
            targets = torch.randint(0, 100, (64,), device=device)

            with record_function("forward_pass"):  # 对前向传播进行标记
                with autocast(enabled=use_amp):
                    outputs = model(inputs)  
                    loss = criterion(outputs, targets) 

            
            with record_function("backward_optimize"):  # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)   
                if scaler is not None:
                    scaler.scale(loss).backward()   
                    scaler.step(optimizer)     
                    scaler.update()       
                else:
                    loss.backward()
                    optimizer.step()

            if (step+1) % 20 == 0:
                print(f"Step [{step+1}/{steps}], Loss: {loss.item():.4f}")
                
            # scheduler.step()
            prof.step()  # 在每个训练步骤后调用，通知Profiler进行记录


    end_time = time.time()
    amp_status = "with" if use_amp else "without"
    print(f"Training time: {end_time - start_time:.2f} seconds, using {amp_status} mixed precision.")


    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_demo.py <True|False>")
        sys.exit(1)
    
    device = torch_cuda_active()
    enable_amp = sys.argv[1].lower() in ['true', '1']
    print(f"Training with mixed precision: {enable_amp}")
    
    model = AlexNet(num_classes=100).to(device)  # 假设num_classes=100
    criterion = nn.CrossEntropyLoss().to(device)  # 保持不变，但确保与你的数据集类别数量匹配
    # optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，降低初始学习率
    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)  # 周期性调整学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler(enabled=enable_amp) if enable_amp else None
    
    summary(model, (3, 224, 224))    #输出模型层信息、输入张量类型信息
    train_loop(model, criterion, optimizer, scaler=scaler, use_amp=enable_amp)
