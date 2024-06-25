# 2024.06.22 训练脚本
#    
# 相比于demo/train.py删除了checkpoint记录
# 之后还需要换更多的数据集，目前只有 CIFAR100
#
# AMP    python train.py -net vgg16 -epoch 5 -enable_amp Ture 
# FP32   python train.py -net vgg16 -epoch 5

import csv
import os
import sys
import argparse
import time
from datetime import datetime
from torchsummary import summary

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, torch_cuda_active
    
scaler = GradScaler(enabled=True)
csv_filename = 'loss.csv'
        
    
def append_info_to_csv(info_data, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([info_data])

def append_to_csv(epoch, loss, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, loss])


def amp_train(epoch):
    
    net.train()
    
    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100)
    
    first_batch = True
    for batch_index, (images, labels) in enumerate(tqdm_bar):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        #适用于 AMP 的训练
        optimizer.zero_grad(set_to_none=True)
        
        if first_batch and epoch == 1:
            with autocast():
                outputs = net.forward_with_print(images)
                loss = loss_function(outputs, labels)
                first_batch = False
        else:
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

    append_to_csv(epoch, running_loss / len(cifar100_training_loader), csv_filename)    
        
    
def fp32_train(epoch):
    
    net.train()

    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100) 
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        if args.gpu:
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

    #写入本次epoch的 loss值
    append_to_csv(epoch, running_loss / len(cifar100_training_loader), csv_filename)   
    
    
@torch.no_grad()
def eval_training(epoch=0, tb=False):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    
    if epoch % 5 == 0:
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s \n'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))
        
        if epoch == settings.EPOCH:
            accuracy_num = correct.float() / len(cifar100_test_loader.dataset)
            append_info_to_csv(round(float(accuracy_num), 4), csv_filename)
        
    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-enable_amp', type=str, default=False, help='if use gpu')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    net = get_network(args)     # 获取到模型
    settings.EPOCH = args.epoch
    if args.enable_amp:
        csv_filename = 'log/' + args.net + '_amp_' + str(args.epoch) + '_' + csv_filename    
    else:
        csv_filename = 'log/' + args.net + '_fp32_' + str(args.epoch) + '_' + csv_filename
    
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    torch_cuda_active()  

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    best_acc = 0.0
    
    if args.enable_amp:
        append_to_csv('Epoch', 'AMP_loss', csv_filename) 
    else:
        append_to_csv('Epoch', 'FP32_loss', csv_filename) 
        
    # summary(net, (3, 224, 224))  # only for alexnet
    summary(net, (3, 32, 32))  
        
    train_start_time = time.time()
    # start training ---------------------------------------------------------------
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.enable_amp:
            amp_train(epoch)
        else:
            fp32_train(epoch)
        
        
        acc = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            best_acc = acc
            continue

    train_end_time = time.time()
    print('{} with {} epoch, enable-AMP {}, Total training time:{:.2f} min'.format(args.net, settings.EPOCH, args.enable_amp, (train_end_time - train_start_time)/60))
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)
    
    

    writer.close()