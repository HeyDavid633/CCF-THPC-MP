# 2024.07.06 CV 类模型的训练脚本 
#
# 数据集针对 imagenet
# 从alexnet开始尝试
#
# AMP    python train_cv_imagenet.py -net alexnet -epoch 5 -precision amp 
# FP32   python train_cv_imagenet.py -net vgg16 -epoch 5 -precision fp32

import csv
import argparse
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from conf import settings
from utils import WarmUpLR, torch_cuda_active, load_ImageNet

scaler = GradScaler(enabled=True)
csv_filename = 'imagenet_loss.csv'
ImageNet_PATH = '/workspace/CCF-THPC-MP/data/imagenet'   # absolute path of working dir
each_epoch_time = []

def append_info_to_csv(info_data, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([info_data])

def append_to_csv(epoch, loss, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, loss])

def fp32_train(epoch): 
    
    net.train()
    running_loss = 0.0
    tqdm_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}', ncols=100)  
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad() # 清零梯度以免累积
        outputs = net(images) # 前向传播 得到预测输出
        loss = loss_function(outputs, labels) # 计算预测输出与真实标签之间的损失
        loss.backward() # 反向 计算梯度
        optimizer.step() # 根据梯度更新模型参数
        running_loss += loss.item() # 累计当前batch的loss到running_loss
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)

        # 学习率预热 --- 为了训练稳定高效
        if epoch <= args.warm:
            warmup_scheduler.step()

    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)
    append_to_csv(epoch, running_loss / len(train_loader), csv_filename)   

def amp_train(epoch):
    net.train()
    running_loss = 0.0
    tqdm_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}', ncols=100)
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = net(images)
            loss = loss_function(outputs, labels)
            
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update() 
        running_loss += loss.item()
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)
    append_to_csv(epoch, running_loss / len(train_loader), csv_filename)    

@torch.no_grad()
def eval_training(epoch=0, tb=False):
    start = timeit.default_timer()
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in val_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = timeit.default_timer()
    
    if epoch % 5 == 0 or settings.EPOCH == epoch:
        print('Evaluating Network: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s \n'.format(
            epoch,
            test_loss / len(val_loader.dataset),
            correct.float() / len(val_loader.dataset),
            finish - start
        ))
        
    if epoch == settings.EPOCH:
        accuracy_num = correct.float() / len(val_loader.dataset)
        append_info_to_csv(round(float(accuracy_num), 4), csv_filename)
        
    return correct.float() / len(val_loader.dataset)


if __name__ == '__main__':
    torch.manual_seed(0)    #确保不同精度在进入权重相同
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-precision', type=str, default=False, help='use which precision: fp16|fp32|amp')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    settings.EPOCH = args.epoch
    
    if args.net == 'alexnet':
        from models.alexnet import alexnet_imagenet
        net = alexnet_imagenet()
    elif args.net == 'vgg16':
        from models.vgg import vgg16_bn_imagenet
        net = vgg16_bn_imagenet()
    
        
    csv_filename = 'log/imagenet/' + args.net + '_' + args.precision + '_' + str(args.epoch) + '_' + csv_filename    
    append_to_csv('Epoch', f'{args.precision}_loss', csv_filename) 
    
    train_loader, val_loader, train_dataset, val_dataset = load_ImageNet(ImageNet_PATH, batch_size = args.batch_size, workers = 4)

    device = torch_cuda_active() 
    net = net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    # summary(net, (3, 224, 224))  
    best_acc = 0.0
    
    train_start_time = timeit.default_timer()
    # start training ---------------------------------------------------------------
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step()

        if args.precision == 'amp':
            amp_train(epoch)
        elif args.precision == 'fp32':
            fp32_train(epoch)
        
        acc = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            best_acc = acc
            continue

    train_end_time = timeit.default_timer()
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)
    
    
    print('\n\nTraining summary', '-'*50,'\n{} with {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(args.net, settings.EPOCH, args.precision, (train_end_time - train_start_time)/60))
    print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc))
    print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
    
    if settings.EPOCH <= 5: 
        summary_info_txt_filename = 'Log_imagenet_Performance_GPU_memory.txt'
    else:
        summary_info_txt_filename = 'Log_imagenet_Accuracy.txt'
        
    with open(summary_info_txt_filename, 'a') as f: 
        print('{} with {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(args.net, settings.EPOCH, args.precision, (train_end_time - train_start_time)/60), file=f)
        print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc), file=f)
        print('Max GPU memory: {:.2f} MB\n'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)), file=f)
        
    
    
    
    
