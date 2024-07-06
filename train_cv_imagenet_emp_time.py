# 2024.07.06 EMP方法 测试指标同 train_cv_imagenet_emp_time.py
# 1epoch时间 
# 100 epoch Accuracy
# GPU显存占用
# 
# emp    python train_cv_imagenet_emp_time.py -net alexnet -epoch 1 -precision emp 
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

csv_filename = 'imagenet_loss.csv'
ImageNet_PATH = '/workspace/CCF-THPC-MP/data/imagenet' 
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
    tqdm_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}', ncols=100)
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        labels = labels.cuda()
        images = images.cuda()

        # EMP 训练1 -------------------------------- 保持以fp32为基准
        optimizer.zero_grad(set_to_none=True)
        outputs = net.forward(images)
        loss = loss_function(outputs, labels)
        loss.backward()                       # 反向传播
        optimizer.step()                      # 根据梯度更新模型参数
        running_loss += loss.item()           # 累计当前batch的loss到running_loss

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
    torch.manual_seed(0)    #确保不同精度在进入权重相同  一般是0  1对于alexnet好收敛
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
      
    model_layer_num = 0
    csv_filename = 'log/' + args.net + '_' + args.precision + '_' + str(args.epoch) + '_' + csv_filename  
    policy_precision_string = ""
    stage1_sampled_info = []
    append_to_csv('Epoch', f'{args.precision}_loss', csv_filename) 
    
    if args.net == 'alexnet':    
        model_layer_num = 20
        policy_precision_string = '0'*model_layer_num
        from models.alexnet import alexnet_emp_imagenet
        net = alexnet_emp_imagenet(policy_precision_string = policy_precision_string)

    elif args.net == 'vgg16':
        model_layer_num = 51
        policy_precision_string = '0'*model_layer_num
        from models.vgg import vgg16_bn_emp_imagenet
        net = vgg16_bn_emp_imagenet(policy_precision_string = policy_precision_string)
        

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
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step()

        emp_train(epoch)
        
        acc = eval_training(epoch)
        
        if best_acc < acc:
            best_acc = acc
            continue

    train_end_time = timeit.default_timer()
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)
    
    print('\n\nTraining summary', '-'*50,'\n{} with {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(args.net, settings.EPOCH, args.precision, (train_end_time - train_start_time)/60))
    print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc))
    print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
    
    if settings.EPOCH <= 5: 
        summary_info_txt_filename = 'Log_Performance_GPU_memory.txt'
    else:
        summary_info_txt_filename = 'Log_Accuracy.txt'
        
    with open(summary_info_txt_filename, 'a') as f: 
        print('{} with {} epoch, \tPrecision Policy: {}, \tTotal training time: {:.2f} min'.format(args.net, settings.EPOCH, args.precision, (train_end_time - train_start_time)/60), file=f)
        print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format(sum(each_epoch_time) / settings.EPOCH, best_acc), file=f)
        print('Max GPU memory: {:.2f} MB\n'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)), file=f)