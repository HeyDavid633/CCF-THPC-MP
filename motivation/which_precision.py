# 6.21 which precision 
# 
# 通过样例程序 分析在PyTorch中到底怎么决定数据精度 
# 如果FP16与FP32同时出现在程序里面，那转换方式与什么时候转换？
# 
# python which_precision.py 
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import torch_cuda_active

# cuda / cpu
device = torch_cuda_active()

x0 = torch.randn(64, 3, 224, 224, device = device, dtype = torch.float32)
weight1 = torch.randn(64, 3, 11, 11, device = device, dtype = torch.float16)
bias1 = torch.randn(64, device = device, dtype = torch.float16)
print("x0 type: ", x0.dtype, "\tdevice: ", x0.device )

# 函数式接口 不能在函数里面操作精度 dtype
# x1 = F.conv2d(x0, weight1, bias1, stride=4, padding=2)
# x1 = x1.half()

# 类式接口 可以更细致的操作，才能支持训练
# 不需要手动传入weight与bias, 但需要指明in_channels与out_channels
# x0是FP16 那么类定义里面也必须写 dtype=torch.half 以指定weight和bias的精度，否则就默认FP32
# layer1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, device=device, dtype=torch.half)
layer1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, device=device)
x1 = layer1(x0)
print("x1 type: ", x1.dtype, "\tdevice: ", x1.device )

x1 = x1.to(torch.float16)

x2 = F.relu(x1)
print("x2 type: ", x2.dtype, "\tdevice: ", x2.device )

x3 = F.max_pool2d(x2, kernel_size=3, stride=2)
print("x3 type: ", x3.dtype, "\tdevice: ", x3.device )