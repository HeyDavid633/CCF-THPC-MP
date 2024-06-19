# 2024.06.12 amp训练脚本  
# 
# 更改了 脚本的入口参数使得更便于我的测试
# 
# 自动混精 python train.py -plat 4080 -net vgg16 -epoch 5 -enable_amp Ture 
# FP32    python train.py -plat 4080 -net vgg16 -epoch 5

import csv
import os
import sys
import argparse
import time
from datetime import datetime

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
delta_csv_filename = 'delta_loss.csv'
        
    
def append_info_to_csv(info_data, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([info_data])

def append_to_csv(epoch, loss, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, loss])

def append_list_to_csv(data_list, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)


def amp_train(epoch):
    
    net.train()
    batch_losses_diff = []  # 新增列表用于记录每个batch损失的增量
    
    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100)
    for batch_index, (images, labels) in enumerate(tqdm_bar):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = net(images)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()  # Scale the loss and call backward
        scaler.step(optimizer)  # Unscales the gradients of optimizer's assigned params in-place
        scaler.update()  # Updates the scale for next iteration.
        running_loss += loss.item()
        
        if batch_index == 0:
            batch_losses_diff.append(0)
        else:
            diff = (running_loss / (batch_index + 1)) - (running_loss / batch_index)
            batch_losses_diff.append(diff)
        
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
            
    epoch_losses_diff = [f"{diff:.4f}" for diff in batch_losses_diff]  
    epoch_info = [epoch] + epoch_losses_diff
    append_list_to_csv(epoch_info, delta_csv_filename) 

    append_to_csv(epoch, running_loss / len(cifar100_training_loader), csv_filename)    
        
    
def fp32_train(epoch):
    
    net.train()
    batch_losses_diff = []  # 新增列表用于记录每个batch损失的增量

    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100) 
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad() 
        outputs = net(images) 
        loss = loss_function(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item() 
        
        if batch_index == 0:
            batch_losses_diff.append(0)
        else:
            diff = (running_loss / (batch_index + 1)) - (running_loss / batch_index)
            batch_losses_diff.append(diff)
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)

        if epoch <= args.warm:
            warmup_scheduler.step()
            
            
    epoch_losses_diff = [f"{diff:.4f}" for diff in batch_losses_diff]  # 转换为字符串以便写入CSV
    epoch_info = [epoch] + epoch_losses_diff
    append_list_to_csv(epoch_info, delta_csv_filename) 

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
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-plat', type=str, required=True, help='running platfrom')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-enable_amp', type=str, default=False, help='if use gpu')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    net = get_network(args)
    settings.EPOCH = args.epoch
    
    if args.enable_amp:
        log_name_info = 'log_' + args.plat + '/' + args.net + '_amp_' + str(args.epoch) + '_'
    else:
        log_name_info = 'log_' + args.plat + '/' + args.net + '_fp32_' + str(args.epoch) + '_'
    csv_filename = log_name_info + csv_filename 
    delta_csv_filename = log_name_info + delta_csv_filename 
    
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
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    

    torch_cuda_active()  # recogonize if GPU is aviliable

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.cuda()
    # writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    
    # 记录csv文件，给出标题
    if args.enable_amp:
        append_to_csv('Epoch', 'AMP_loss', csv_filename) 
    else:
        append_to_csv('Epoch', 'FP32_loss', csv_filename) 
        
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
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            
    train_end_time = time.time()
    print('{} with {} epoch, enable-AMP {}, Total training time:{:.2f} min'.format(args.net, settings.EPOCH, args.enable_amp, (train_end_time - train_start_time)/60))
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)

    # writer.close()