# 6.19 训练的demo2 以vgg16为例
# 
# 构建简单的模型，在其中试验数据类型的改变 -- 强制改变 dtype  
# 构建了AlexNet() 没有拆开 但训练不收敛，修改了AlexNet的部分维度信息
# 
# python train_demo2.py False
# 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import profile, record_function, ProfilerActivity
from torchsummary import summary
from datetime import datetime
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
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = x.view(x.size(0), -1)  # 自适应池化后，直接展平
        x = self.classifier(x)
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

            if (step+1) % 10 == 0:
                print(f"Step [{step+1}/{steps}], Loss: {loss.item():.4f}")
                
            prof.step()  # 在每个训练步骤后调用，通知Profiler进行记录


    end_time = time.time()
    amp_status = "with" if use_amp else "without"
    print(f"Training time: {end_time - start_time:.2f} seconds, using {amp_status} mixed precision.")



def main(enable_amp):
    model = AlexNet(num_classes=100).to(device)  # 假设num_classes=100
    criterion = nn.CrossEntropyLoss().to(device)  # 保持不变，但确保与你的数据集类别数量匹配
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler(enabled=enable_amp) if enable_amp else None
    
    summary(model, (3, 32, 32))    #输出模型层信息、输入张量类型信息
    train_loop(model, criterion, optimizer, scaler=scaler, use_amp=enable_amp)
    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_demo.py <True|False>")
        sys.exit(1)
    
    device = torch_cuda_active()
    enable_amp = sys.argv[1].lower() in ['true', '1']
    print(f"Training with mixed precision: {enable_amp}")
    main(enable_amp)
