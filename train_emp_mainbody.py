# 2024.07.05 EMP方法全流程的脚本 
# 专门测breakdown  
#
# 数据集针对 CIFAR100 以AlexNet为例实现主体方法 :
# 模型训练的信息提取， 需要实现特殊的训练器1-按epoch采集、训练器2-按batch采集
# 
# emp    python train_emp_mainbody.py -net alexnet -epoch 1 -precision emp 
import csv
import argparse
import timeit
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, torch_cuda_active
    
summary_info_txt_filename = '0705.txt'
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
    
def fp32_train(epoch):
    net_fp32.train()
    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100) 
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()                   # 清零梯度以免累积
        outputs = net_fp32(images)              # 前向传播 得到预测输出
        loss = loss_function(outputs, labels)   # 计算预测输出与真实标签之间的损失
        loss.backward()                         # 反向 计算梯度
        optimizer.step()                        # 根据梯度更新模型参数
        running_loss += loss.item()             # 累计当前batch的loss到running_loss
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)

        if epoch <= args.warm:
            warmup_scheduler.step()
            
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time

    return running_loss / iter_per_epoch, epoch_elapsed_time
    
def emp_train(epoch):
    net.train()
    running_loss = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Training Epoch {epoch}', ncols=100)
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
    append_to_csv(epoch, running_loss / len(cifar100_training_loader), csv_filename)

def stage1_sample(sampling_epoch):
    net.train()
    running_loss_this_epoch = 0.0
    tqdm_bar = tqdm(cifar100_training_loader, desc=f'Sampling {sampling_epoch}', ncols=100)
    epoch_start_time = timeit.default_timer()
    
    for batch_index, (images, labels) in enumerate(tqdm_bar):
        
        labels = labels.cuda()
        images = images.cuda()
    
        optimizer.zero_grad()                   # 清零梯度以免累积
        outputs = net(images)                   # 前向过程
        loss = loss_function(outputs, labels)   # 计算预测输出与真实标签之间的损失
        loss.backward()                         # 反向 计算梯度
        optimizer.step()                        # 根据梯度更新模型参数
        running_loss_this_epoch += loss.item()  # 累计当前batch的loss到running_loss
        
        postfix = {'Loss': f"{running_loss_this_epoch / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)
        
        warmup_scheduler.step()      # 认为这个阶段每个epoch都是第一次进来
    
    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    
    return running_loss_this_epoch / iter_per_epoch, epoch_elapsed_time
    
def stage2_sample():
    net.train()
    batch_start_time = timeit.default_timer()
    running_loss_this_batch = 0.0
     
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        
        labels = labels.cuda()
        images = images.cuda()
    
        optimizer.zero_grad()                   # 清零梯度以免累积
        outputs = net(images)                   # 前向过程
        loss = loss_function(outputs, labels)   # 计算预测输出与真实标签之间的损失
        loss.backward()                         # 反向 计算梯度
        optimizer.step()                        # 根据梯度更新模型参数
        running_loss_this_batch += loss.item()  # 累计当前batch的loss到running_loss
        
        if outputs is not None: break

    torch.cuda.synchronize()
    batch_elapsed_time = timeit.default_timer() - batch_start_time
    
    return batch_elapsed_time

def info_sample():
    net.train()    
    stage1_op, all_layer_op = [], []
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        labels = labels.cuda()
        images = images.cuda()
    
        stage1_op, all_layer_op, outputs = net.forward_with_print(images) 
                
        if outputs is not None: break
    
    return stage1_op, all_layer_op
            
def generate_strings_for_stage2(stage1_policy_selected, stage1_op):
    # 添加两端的边界，使得算法更简洁
    stage1_op.insert(0, {'id': -1})
    stage1_op.append({'id': len(stage1_policy_selected) - 1})  # 修改这里
    
    # 初始化结果列表
    results = [stage1_policy_selected]
    
    # 遍历每一个分割点
    for i in range(len(stage1_op) - 1):
        start = stage1_op[i]['id'] + 1
        end = stage1_op[i + 1]['id']  # 不需要修改这里，但需要确保在后续逻辑中正确使用
        
        # 只有当段的长度大于1且开始和结束字符不同才需要枚举
        if end - start > 0 and stage1_policy_selected[start] != stage1_policy_selected[end]:
            # 保存当前的结果数量，用于后续添加新结果
            prev_results_len = len(results)
            
            # 枚举这一段的所有二进制组合
            for num in range(2**(end - start )):  
                binary_str = format(num, '0' + str(end - start) + 'b')  
                
                # 对每一个已有结果进行修改，生成新的结果
                for j in range(prev_results_len):
                    new_str = list(results[j])
                    for k, bit in enumerate(binary_str):
                        new_str[start + k] = bit
                    results.append(''.join(new_str))
                    
    return results[1:]  # 去除第一个原始字符串


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
            if args.precision == 'fp16':
                images = images.half()

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
        
    return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':
    torch.manual_seed(1)    #确保不同精度在进入权重相同  一般是0  1对于alexnet好收敛
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

    
    if args.net == 'alexnet':    
        model_layer_num = 20
        policy_precision_string = '0'*model_layer_num
        adjustable_id_list = [0, 14]
        from models.alexnet import alexnet_emp, alexnet
        net = alexnet_emp(policy_precision_string = policy_precision_string)
        net_fp32 = alexnet()
    elif args.net == 'vgg16':
        model_layer_num = 51
        policy_precision_string = '0'*model_layer_num
        adjustable_id_list = [0, 14]
        from models.vgg import vgg16_bn_emp
        net = vgg16_bn_emp(policy_precision_string = policy_precision_string)
                

    device = torch_cuda_active() 
    loss_function = nn.CrossEntropyLoss()
    iter_per_epoch = len(cifar100_training_loader)


    stage1_start_time = timeit.default_timer()
    
    # Stage1: To get best-speed policy with acceptable loss ---------------------------
    net_fp32 = net_fp32.to(device)   
    optimizer = optim.SGD(net_fp32.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    policy_precision_string = '1'*model_layer_num
    running_loss_this_epoch, epoch_time = fp32_train(1)
    stage1_sampled_info.append({"id":0, "policy_precision":policy_precision_string, "loss":running_loss_this_epoch, "time":epoch_time})
    
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    policy_precision_string = '0'*model_layer_num
    running_loss_this_epoch, epoch_time = stage1_sample(0)  # 默认policy的情况下就应跑一个
    stage1_sampled_info.append({"id":0, "policy_precision":policy_precision_string, "loss":running_loss_this_epoch, "time":epoch_time})
    
    stage1_op, all_layer_op = info_sample()   
    # 用一个batch 采集算子各个层的精度信息, 检查对于这个串的采集结果 --- 两个串（实际上adjustable_id_list也应在这里体现）            
    print("\nStage 1 Operations "+"-"*50)
    for op in stage1_op: print(f"{op['id']:3d}\t{op['layer_name']:10s}\t{op['data_type']}")
    print("\nAll Layer Operations "+"-"*50)
    for op in all_layer_op: print(f"{op['id']:3d}\t{op['layer_name']:10s}\t{op['data_type']}")
    
    stage1_sampler_counter = 0
    for op in stage1_op:
        policy_precision_string = '0' * model_layer_num
        if op['id'] in adjustable_id_list:
            index_to_change = op['id']
            policy_precision_string = policy_precision_string[:index_to_change] + '1' + policy_precision_string[index_to_change+1:]
            
            # 模型的信息刷新 -- 不同精度策略模型不同了
            net = alexnet_emp(policy_precision_string = policy_precision_string)
            net = net.to(device)
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
            
            stage1_sampler_counter = stage1_sampler_counter + 1
            running_loss_this_epoch, epoch_time = stage1_sample(stage1_sampler_counter)
            stage1_sampled_info.append({"id":index_to_change, "policy_precision":policy_precision_string, "loss":running_loss_this_epoch, "time":epoch_time})
            
            # 当前模型的各层信息
            # temp1, all_layer_op = info_sample()  
            # print("All Layer Operations "+"-"*50)
            # for op in all_layer_op: 
            #     print(f"{op['id']:3d}\t{op['layer_name']:15s}\t{op['data_type']}")
    
    
    print("\n", "-"*50 ," Stage1 sample info")
    for sampler_info in stage1_sampled_info:
        print(f"{sampler_info['id']:3d}\t{sampler_info['policy_precision']:10s}\t{sampler_info['loss']:.4f}\t{sampler_info['time']:.2f} s")
    
    threshold_loss = 1.02
    stage1_policy_selected = stage1_sampled_info[0]['policy_precision']
    best_speed = stage1_sampled_info[0]['time']
    fp32_loss = stage1_sampled_info[0]['loss']
    for i in range(1, len(stage1_sampled_info)):
        if stage1_sampled_info[i]['loss'] < threshold_loss * fp32_loss and stage1_sampled_info[i]['time'] < best_speed:
            best_speed = stage1_sampled_info[i]['time']
            stage1_policy_selected = stage1_sampled_info[i]['policy_precision']
            
    print("Stage1 policy selected: ", f"{stage1_policy_selected:10s}\t{best_speed:.2f} s") 
    torch.cuda.synchronize()
    stage1_elapsed_time = timeit.default_timer() - stage1_start_time
    
    
    stage2_start_time = timeit.default_timer()
    # Bridge between Stage1 and Stage2 : Generate search space of Stage2 ---------------------------
    print("\n", "-"*50 ," Generate search space of Stage2")
    strings_for_stage2 = generate_strings_for_stage2(stage1_policy_selected, stage1_op)
    print("len(strings_for_stage2): ", len(strings_for_stage2), "  ", strings_for_stage2)
    
    
    
    # Stage2: To get best-speed policy with acceptable loss ---------------------------
    stage2_sampler_counter = 0
    stage2_policy_selected = stage1_policy_selected
    net = alexnet_emp(policy_precision_string = stage2_policy_selected)
    net = net.to(device)
    best_speed = stage2_sample()
    if len(strings_for_stage2) > 1:
        for stage2_policy_precision_string in strings_for_stage2[1:]:
            net = alexnet_emp(policy_precision_string = stage2_policy_precision_string)
            net = net.to(device)
            batch_time = stage2_sample()
        
            if batch_time < best_speed:
                best_speed = batch_time
                stage2_policy_selected = stage2_policy_precision_string

    print("\n", "-"*50 ," Stage2 sample info")
    print("Stage2 policy selected: ", f"{stage2_policy_selected:10s}\t{best_speed:.2f} s")  
    stage2_elapsed_time = timeit.default_timer() - stage2_start_time
    
    
    # Chose the best policy and trian with EMP
    best_acc = 0.0
    
    net = alexnet_emp(policy_precision_string = stage2_policy_selected)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) 
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
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
    
    print('For time breakdown\tStage1 Cost: {:.2f} sec, Stage2 Cost: {:.2f} sec'.format(stage1_elapsed_time, stage2_elapsed_time))