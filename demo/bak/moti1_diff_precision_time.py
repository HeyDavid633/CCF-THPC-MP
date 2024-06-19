# 6.19 AlexNet前向过程 --- 完全展开
#
# 1. 只保留AlexNet的前向过程，对各个算子执行时间进行计时 
# 2. 张量在中间强制转换一波
#
import sys
import time
import torch
import torch.nn.functional as F
from utils import torch_cuda_active

def alexnet_forward(x, conv_params, fc_params, use_relu=True):
    # self.features
    x = F.conv2d(x, conv_params[0][0], bias=conv_params[0][1], stride=4, padding=2)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = F.conv2d(x, conv_params[1][0], bias=conv_params[1][1], padding=2)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = F.conv2d(x, conv_params[2][0], bias=conv_params[2][1], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_params[3][0], bias=conv_params[3][1], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, conv_params[4][0], bias=conv_params[4][1], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    
    # 调整形状以便进入全连接层, 同 x = x.view(x.size(0), 256 * 6 * 6)
    x = x.view(x.size(0), -1)
    
    # self.classifier
    x = F.linear(x, fc_params[0][0], bias=fc_params[0][1])
    x = F.relu(x)
    x = F.dropout(x, training=True)  # 注意，此处的dropout在训练时才启用
    x = F.linear(x, fc_params[1][0], bias=fc_params[1][1])
    x = F.relu(x)
    x = F.dropout(x, training=True)
    x = F.linear(x, fc_params[2][0], bias=fc_params[2][1])
    
    return x


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python moti1_diff_precision_time.py FP16|FP32")
        sys.exit(1)
    data_type = sys.argv[1]
    print(f"Training with mixed precision: {data_type}")   
    
    
    device = torch_cuda_active()
    exec_times = 1
    x_example = torch.randn(64, 3, 224, 224, device=device)
    # 示例：构建权重和偏置的参数列表，这里只是构造了一些随机张量作为示例
    conv_params_example = [
        (torch.randn(64, 3, 11, 11, device=device), torch.randn(64, device=device)),  # 示例卷积层参数
        (torch.randn(192, 64, 5, 5, device=device), torch.randn(192, device=device)),
        (torch.randn(384, 192, 3, 3, device=device), torch.randn(384, device=device)),
        (torch.randn(256, 384, 3, 3, device=device), torch.randn(256, device=device)),
        (torch.randn(256, 256, 3, 3, device=device), torch.randn(256, device=device))
    ]

    fc_params_example = [
        (torch.randn(4096, 256 * 6 * 6, device=device), torch.randn(4096, device=device)),
        (torch.randn(4096, 4096, device=device), torch.randn(4096, device=device)),
        (torch.randn(1000, 4096, device=device), torch.randn(1000, device=device))  
    ]


    total_start_time = time.time()
    for i in range(exec_times):
        output = alexnet_forward(x_example, conv_params_example, fc_params_example)
    total_finish_time = time.time()
    
    print(f"Running {exec_times} times Cost: {total_finish_time - total_start_time:.2f} seconds, using {data_type}")
