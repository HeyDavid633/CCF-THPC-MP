# 2024.07.03 主体方法实现 
#
# 数据集针对 CIFAR100
# 只折腾alexnet 工程实现  所以只保留EMP的实现
#
# EMP    python train_cv_test.py -net alexnet -epoch 5

import csv
import argparse
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary
from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, torch_cuda_active
    
device = torch_cuda_active() 
scaler = GradScaler(enabled=True)
csv_filename = 'loss.csv'
each_epoch_time = []
        
def append_info_to_csv(info_data, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([info_data])

def append_to_csv(epoch, loss, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, loss])
    
def emp_train(epoch):
    
    net.train()
    
    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100)
    
    first_batch = True
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        # EMP 训练1 -------------------------------- 保持以fp32为基准
        optimizer.zero_grad(set_to_none=True)
        
        if first_batch and epoch == 1:
            outputs = net.forward_with_print(images)
            loss = loss_function(outputs, labels)
            first_batch = False
        else:
            outputs = net.forward(images)
            loss = loss_function(outputs, labels)
            
        loss.backward()                       # 反向传播
        optimizer.step()                      # 根据梯度更新模型参数
        running_loss += loss.item()           # 累计当前batch的loss到running_loss
        
        
        
        # EMP 训练2 ------------------------------- 开启梯度缩放的保护
        # optimizer.zero_grad(set_to_none=True)

        # if first_batch and epoch == 1:
        #     outputs = net.forward_with_print(images)
        #     loss = loss_function(outputs, labels)
        #     first_batch = False
        # else:
        #     outputs = net(images)
        #     loss = loss_function(outputs, labels)
            
        # scaler.scale(loss).backward()  
        # scaler.step(optimizer)  
        # scaler.update() 
        # running_loss += loss.item()
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
        if epoch <= args.warm:
            warmup_scheduler.step()
            
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)
    append_to_csv(epoch, running_loss / len(cifar100_training_loader), csv_filename)   

    
@torch.no_grad()
def eval_training(epoch=0, tb=False):

    start = timeit.default_timer()
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

    finish = timeit.default_timer()
    
    if epoch % 5 == 0 or settings.EPOCH == epoch:
        print('Evaluating Network: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s \n'.format(
            epoch,
            test_loss / len(cifar100_test_loader.dataset),
            correct.float() / len(cifar100_test_loader.dataset),
            finish - start
        ))
        
    if epoch == settings.EPOCH:
        accuracy_num = correct.float() / len(cifar100_test_loader.dataset)
        append_info_to_csv(round(float(accuracy_num), 4), csv_filename)
        final_accuracy = accuracy_num
        

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    torch.manual_seed(0)    #确保不同精度在进入权重相同
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
    settings.EPOCH = args.epoch
   
    csv_filename = 'log/' + args.net + '_test_' + str(args.epoch) + '_' + csv_filename    
    
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
    
    
    policy_precision_string = '00000000000000000000'
    if args.net == 'alexnet':   
        from models.alexnet_test import alexnet
        net = alexnet(policy_precision_string = policy_precision_string)
    
    net = net.to(device)
    
    summary(net, (3, 32, 32)) 

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm) 

    best_acc = 0.0
    
    append_to_csv('Epoch', 'EMP_loss', csv_filename) 
    
    train_start_time = timeit.default_timer()
    # start training ---------------------------------------------------------------
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step()
            
        emp_train(epoch)
        
        acc = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            best_acc = acc
            continue

    train_end_time = timeit.default_timer()
    final_accuracy, final_loss = 0.0, 0.0
    print('\n\nTraining summary', '-'*50,'\n{} with {} epoch, Precision Policy: EMP, Total training time: {:.2f} min'.format(args.net, settings.EPOCH, (train_end_time - train_start_time)/60))
    print('Each epoch average cost time {:.2f} sec\t\t Final Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH), final_accuracy)
    
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)
    
