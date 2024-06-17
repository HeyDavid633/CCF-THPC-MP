# 6.17 训练的demo
# 构建了简单的 4层的全连接 网络
#   
# 1. 熟悉PyTorch的模型定义过程        OK
# 2. 模型结构信息 输入输出尺寸信息输出   OK
# 3. 统计各算子的执行时间 - tensorboard   
# 
# python train_demo.py False
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


class CustomSimpleNet(nn.Module):
    def __init__(self, input_dim=128*128*3, hidden_dims=[128, 64, 128], output_dim=10):
        super(CustomSimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], output_dim)
        
    def forward(self, x):   # 可以传入一个策略 dtype信息 --
        x = torch.flatten(x, 1)          
        x = torch.relu(self.fc1(x))  # dtype()
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x)) # 也可以写作 x = torch.matmul(x, self.fc3.weight.t()) + self.fc3.bias
        x = torch.relu(x)
        x = self.fc4(x)
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
            inputs = torch.randn(64, 3, 128, 128, device=device)
            targets = torch.randint(0, 10, (64,), device=device)

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
    model = CustomSimpleNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler(enabled=enable_amp) if enable_amp else None
    
    summary(model, (3, 128, 128))    #输出模型层信息、输入张量类型信息
    train_loop(model, criterion, optimizer, scaler=scaler, use_amp=enable_amp)
    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_demo.py <True|False>")
        sys.exit(1)
    
    device = torch_cuda_active()
    enable_amp = sys.argv[1].lower() in ['true', '1']
    print(f"Training with mixed precision: {enable_amp}")
    main(enable_amp)
