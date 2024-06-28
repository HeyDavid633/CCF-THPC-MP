# 6.19 AlexNet前向过程 --- 完全展开
#
# 1. 只保留AlexNet的前向过程，对各个算子执行时间进行计时 
# 2. 张量在中间强制转换一波
#
# python moti1_diff_precision_time.py FP32 | FP16
import sys
import timeit
# import time
import torch
import torch.nn.functional as F
from datetime import datetime
import csv
from utils import torch_cuda_active  

class OperatorInfo:
    def __init__(self, name, input_shape, dtype):
        self.name = name
        self.input_shape = input_shape
        self.dtype = dtype
        self.total_time = 0.0
        
def write_to_csv(operator_infos, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["op_name", "input_shape", "exec_dtype", "avg_exec_time"])
        for info in operator_infos:
            writer.writerow([info.name, info.input_shape, info.dtype, info.avg_exec_time])

def alexnet_forward(x, conv_params, fc_params, use_relu=True):
    operator_infos = []

    # self.features
    operator_infos.append(OperatorInfo("conv2d-1", x.shape, x.dtype))
    x = F.conv2d(x, conv_params[0][0], bias=conv_params[0][1], stride=4, padding=2)
    operator_infos.append(OperatorInfo("relu-2", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("max_pool2d-3", x.shape, x.dtype))
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    operator_infos.append(OperatorInfo("conv2d-4", x.shape, x.dtype))
    x = F.conv2d(x, conv_params[1][0], bias=conv_params[1][1], padding=2)
    operator_infos.append(OperatorInfo("relu-5", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("max_pool2d-6", x.shape, x.dtype))
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    operator_infos.append(OperatorInfo("conv2d-7", x.shape, x.dtype))
    x = F.conv2d(x, conv_params[2][0], bias=conv_params[2][1], padding=1)
    operator_infos.append(OperatorInfo("relu-8", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("conv2d-9", x.shape, x.dtype))
    x = F.conv2d(x, conv_params[3][0], bias=conv_params[3][1], padding=1)
    operator_infos.append(OperatorInfo("relu-10", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("conv2d-11", x.shape, x.dtype))
    x = F.conv2d(x, conv_params[4][0], bias=conv_params[4][1], padding=1)
    operator_infos.append(OperatorInfo("relu-12", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("max_pool2d-13", x.shape, x.dtype))
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    
    # 调整形状以便进入全连接层, 同 x = x.view(x.size(0), 256 * 6 * 6)
    x = x.view(x.size(0), -1)
    
    # self.classifier
    operator_infos.append(OperatorInfo("linear-14", x.shape, x.dtype))
    x = F.linear(x, fc_params[0][0], bias=fc_params[0][1])
    operator_infos.append(OperatorInfo("relu-15", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("dropout-16", x.shape, x.dtype))
    x = F.dropout(x, training=True)  # 注意，此处的dropout在训练时才启用
    operator_infos.append(OperatorInfo("linear-17", x.shape, x.dtype))
    x = F.linear(x, fc_params[1][0], bias=fc_params[1][1])
    operator_infos.append(OperatorInfo("relu-18", x.shape, x.dtype))
    x = F.relu(x)
    operator_infos.append(OperatorInfo("dropout-19", x.shape, x.dtype))
    x = F.dropout(x, training=True)
    operator_infos.append(OperatorInfo("linear-20", x.shape, x.dtype))
    x = F.linear(x, fc_params[2][0], bias=fc_params[2][1])
    
    
    for _ in range(exec_times):
        
        x = torch.randn(64, 3, 224, 224, device=device, dtype=data_type)
        
        
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        x = F.conv2d(x, conv_params[0][0], bias=conv_params[0][1], stride=4, padding=2)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[0].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[1].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[2].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.conv2d(x, conv_params[1][0], bias=conv_params[1][1], padding=2)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[3].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[4].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[5].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.conv2d(x, conv_params[2][0], bias=conv_params[2][1], padding=1)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[6].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[7].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.conv2d(x, conv_params[3][0], bias=conv_params[3][1], padding=1)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[8].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[9].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.conv2d(x, conv_params[4][0], bias=conv_params[4][1], padding=1)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[10].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[11].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[12].total_time += elapsed_time
        
        
        x = x.view(x.size(0), -1)
        
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        x = F.linear(x, fc_params[0][0], bias=fc_params[0][1])
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[13].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[14].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.dropout(x, training=True) 
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[15].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.linear(x, fc_params[1][0], bias=fc_params[1][1])
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[16].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.relu(x)
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[17].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.dropout(x, training=True)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[18].total_time += elapsed_time
        
        start_time = timeit.default_timer()
        x = F.linear(x, fc_params[2][0], bias=fc_params[2][1])
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start_time
        operator_infos[19].total_time += elapsed_time
        
        
    
    for info in operator_infos:
        info.avg_exec_time = info.total_time * 1000 / exec_times  # 转换为毫秒    
    
    
    return operator_infos


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python moti1_diff_precision_time.py BF16|FP16|FP32")
        sys.exit(1)
    if sys.argv[1] == 'FP32':
        data_type = torch.float32
    elif sys.argv[1] == 'FP16':
        data_type = torch.float16
    elif sys.argv[1] == 'BF16':
        data_type = torch.bfloat16
    elif sys.argv[1] == 'FP8e4m3':
        data_type = torch.float8_e4m3fn
    elif sys.argv[1] == 'FP8e5m2':
        data_type = torch.float8_e5m2
        
    print(f"Training with mixed precision: {data_type}")   
    
    
    device = torch_cuda_active()
    exec_times = 200
    x_example = torch.randn(64, 3, 224, 224, device=device, dtype=data_type)
    # 示例：构建权重和偏置的参数列表，这里只是构造了一些随机张量作为示例
    conv_params_example = [
        (torch.randn(64, 3, 11, 11, device=device, dtype=data_type), torch.randn(64, device=device, dtype=data_type)),  # 示例卷积层参数
        (torch.randn(192, 64, 5, 5, device=device, dtype=data_type), torch.randn(192, device=device, dtype=data_type)),
        (torch.randn(384, 192, 3, 3, device=device, dtype=data_type), torch.randn(384, device=device, dtype=data_type)),
        (torch.randn(256, 384, 3, 3, device=device, dtype=data_type), torch.randn(256, device=device, dtype=data_type)),
        (torch.randn(256, 256, 3, 3, device=device, dtype=data_type), torch.randn(256, device=device, dtype=data_type))
    ]

    fc_params_example = [
        (torch.randn(4096, 256 * 6 * 6, device=device, dtype=data_type), torch.randn(4096, device=device, dtype=data_type)),
        (torch.randn(4096, 4096, device=device, dtype=data_type), torch.randn(4096, device=device, dtype=data_type)),
        (torch.randn(1000, 4096, device=device, dtype=data_type), torch.randn(1000, device=device, dtype=data_type))  
    ]

    total_start_time = timeit.default_timer()
    operator_infos = alexnet_forward(x_example, conv_params_example, fc_params_example)
    total_finish_time = timeit.default_timer()
    
    now = datetime.now().strftime('%b%d_%H_%M_%S')
    filename = f"table1_{sys.argv[1]}_{now}.csv"
    write_to_csv(operator_infos, filename)
    
    print(f"Running {exec_times} times Cost: {total_finish_time - total_start_time:.2f} seconds, using {data_type}")
