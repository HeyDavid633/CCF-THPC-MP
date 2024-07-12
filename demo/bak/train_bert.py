# 2024.07.02 bert模型训练脚本 
#
# 数据集针对 SQuAD 
# 先通过 squad_feature_creation.py生成特征文件 这里进行调用 --- 也没成功
# 训练OK，但模型性能评估还有一定的问题 退一步不要EM/F1 只要一个Accuracy
#
# AMP    python train_bert.py -epoch 5 -batch_size 16 -precision amp 
# FP32   python train_bert.py -epoch 5 -precision fp32

import csv
import argparse
import timeit
from tqdm import tqdm
from utils import torch_cuda_active
import pickle
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from transformers.data.processors.squad import SquadV2Processor


scaler = GradScaler(enabled=True)
each_epoch_time = []
csv_filename = 'loss.csv'

device = torch_cuda_active()

def append_info_to_csv(info_data, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([info_data])

def append_to_csv(epoch, loss, filename):
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, loss])

def fp32_train(epoch):
    model.train()
    running_loss = 0.0
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch}', ncols=100)
    
    epoch_start_time = timeit.default_timer()
    
    for batch_index, batch in enumerate(tqdm_bar):
        
        optimizer.zero_grad()
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(device) for t in batch)
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, 
                        start_positions=start_positions, 
                        end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)

    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)
    append_to_csv(epoch, running_loss / len(train_dataloader), csv_filename) 

def amp_train(epoch):
    model.train()
    running_loss = 0.0
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch}', ncols=100)
    
    epoch_start_time = timeit.default_timer()
    
    for batch_index, batch in enumerate(tqdm_bar):
        
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(device) for t in batch)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast:
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            start_positions=start_positions, 
                            end_positions=end_positions)
            loss = outputs.loss
            
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update() 
        running_loss += loss.item()
        
        postfix = {'Loss': f"{running_loss / (batch_index + 1):.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.4f}"}
        tqdm_bar.set_postfix(**postfix)

    torch.cuda.synchronize()
    epoch_elapsed_time = timeit.default_timer() - epoch_start_time
    each_epoch_time.append(epoch_elapsed_time)
    append_to_csv(epoch, running_loss / len(train_dataloader), csv_filename) 

if __name__ == '__main__':
    torch.manual_seed(0)    #确保不同精度在进入权重相同
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=5, help='epoch to train')
    parser.add_argument('-precision', type=str, default=False, help='use which precision: fp16|fp32|amp')
    parser.add_argument('-gpu', type=str, default=True, help='if use gpu')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()
     
    train_batch_size = args.batch_size
    num_epochs = args.epoch
    learning_rate = 3e-5
    csv_filename = 'log/BERT/BERT_' + args.precision + '_' + str(args.epoch) + '_' + csv_filename   
    
    # 加载SQuAD 2.0数据集的特征
    with open('data/SQuAD/train_features.pkl', 'rb') as f:
        train_features = pickle.load(f)

    # 将特征转换为PyTorch张量
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions)
    num_samples = 100
    train_dataset = TensorDataset(
        all_input_ids[:num_samples], 
        all_attention_mask[:num_samples], 
        all_token_type_ids[:num_samples], 
        all_start_positions[:num_samples], 
        all_end_positions[:num_samples])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    
    # 加载BERT模型和优化器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    append_to_csv('Epoch', f'{args.precision}_loss', csv_filename) 
    train_start_time = timeit.default_timer()
    # 微调BERT ------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        
        if args.precision == 'fp32':
            fp32_train(epoch)
        elif args.precision == 'amp':
            amp_train(epoch)
    
    train_end_time = timeit.default_timer()
    print('\n\nTraining summary', '-'*50,'\nBERT-base with {} epoch, Precision Policy: {}, Total training time: {:.2f} min'.format( num_epochs, args.precision, (train_end_time - train_start_time)/60))
    print('Each epoch average cost time {:.2f} sec'.format(sum(each_epoch_time) / num_epochs))
    append_info_to_csv(round((train_end_time - train_start_time)/60, 2), csv_filename)
     

    # print("Model evaluting ... ...")
    # accuracy = evaluate_model()
    # print(f"Accuracy: {accuracy}%")

    # 保存微调后的模型
    model.save_pretrained("data/SQuAD/SQuAD_finetuned_bert")