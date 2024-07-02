# 2024.07.01 跑模型的训练脚本备份  
#
# 数据集针对 SQuAD 
# 一天半尝试后 仍未解决SQuAD测评的问题 即eval_training()中的格式反复对不上
# 退一步 --- 只要可以输出Accuracy就可以
#
# AMP    python train_bert.py -epoch 5 -batch_size 16 -precision amp 
# FP32   python train_bert.py -epoch 5 -precision fp32 -batch_size 16

import csv
import os
import sys
import argparse
import timeit
from tqdm import tqdm
from datetime import datetime
import numpy as np
# from torchsummary import summary
from utils import torch_cuda_active
import pickle
from transformers import BertForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, AdamW, EvalPrediction
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from datasets import load_metric

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

# 评估未经微调的BERT的性能
def china_capital():
    question, text = "What is the capital of China?", "The capital of China is Beijing."
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    predict_answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
    predicted_answer = tokenizer.decode(predict_answer_tokens)
    print("中国的首都是？", predicted_answer)   

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

from transformers import BertForQuestionAnswering, BertTokenizerFast
from datasets import load_metric
import numpy as np
import torch
from tqdm.auto import tqdm

# 加载模型和分词器
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
metric = load_metric("squad_v2")

def get_predictions(features, logits):
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    
    # 获取预测的起始和结束位置
    start_indices = np.argmax(start_logits, axis=1)
    end_indices = np.argmax(end_logits, axis=1)
    
    predictions = []
    for feature, (start_index, end_index) in zip(features, zip(start_indices, end_indices)):
        # 使用token到原文的映射来恢复答案
        orig_answer_text = " ".join(feature.tokens[start_index:(end_index + 1)])
        # 注意：这里的逻辑简化了，实际应用中需要考虑token映射到原始文本的偏移
        context = feature.context_text
        offset_mapping = feature.offset_mapping  # 假设特征中包含offset_mapping
        
        # 应用偏移量还原答案到原文
        start_char = offset_mapping[start_index][0]
        end_char = offset_mapping[end_index][1]
        answer_text = context[start_char:end_char]
        
        predictions.append({
            "id": feature.example_id,
            "prediction_text": answer_text,
            "answer_start": start_char  # 基于字符偏移量
        })
    return predictions

@torch.no_grad()
def eval_training(epoch):
    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in tqdm(val_dataloader, desc="Evaluating"):
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device) if len(batch) > 2 else None,
        }
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        
        all_start_logits.append(start_logits)
        all_end_logits.append(end_logits)
    
    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)
    logits = np.stack((all_start_logits, all_end_logits), axis=-1)

    # 加载原始数据的特征和元数据
    with open('data/SQuAD/train_features.pkl', 'rb') as f:
        all_features = pickle.load(f)
    val_features = all_features[:int(len(all_features)*0.2)]  # 示例分割逻辑
    
    # 从logits中得到预测的答案
    predictions = get_predictions(val_features, logits)
    
    # 加载原始数据并调整格式以匹配评估标准
    with open('data/SQuAD/train_original_data.pkl', 'rb') as f:
        original_data = pickle.load(f)
    val_original_data = original_data[:int(len(original_data)*0.2)]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_original_data]

    results = metric.compute(predictions=predictions, references=references)
    em = results["exact"]
    f1 = results["f1"]
    print(f"Validation Results - Exact Match: {em:.4f}, F1: {f1:.4f}")


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
        
    # 假设已经加载了train_features和original_data
    train_features = pickle.load(open('data/SQuAD/train_features.pkl', 'rb'))
    original_data = pickle.load(open('data/SQuAD/train_original_data.pkl', 'rb'))

    # 对数据进行分割
    num_samples = 200
    # num_samples = len(train_features)
    split_index = int(num_samples * 0.8)  # 80%数据用于训练，20%用于验证
    train_features, val_features = train_features[:split_index], train_features[split_index:]
    train_original, val_original = original_data[:split_index], original_data[split_index:]

    # 提取必要的数据以创建TensorDataset 
    train_input_ids, train_attention_masks, train_token_type_ids, train_start_positions, train_end_positions = zip(*[(feature.input_ids, feature.attention_mask, feature.token_type_ids, feature.start_position, feature.end_position) for feature in train_features])
    val_input_ids, val_attention_masks, val_token_type_ids, val_start_positions, val_end_positions = zip(*[(feature.input_ids, feature.attention_mask, feature.token_type_ids, feature.start_position, feature.end_position) for feature in val_features])
    # 将zip对象转换为numpy数组以保持与TensorDataset兼容
    train_input_ids = np.array(train_input_ids)
    train_attention_masks = np.array(train_attention_masks)
    train_token_type_ids = np.array(train_token_type_ids)
    train_start_positions = np.array(train_start_positions)
    train_end_positions = np.array(train_end_positions)

    val_input_ids = np.array(val_input_ids)
    val_attention_masks = np.array(val_attention_masks)
    val_token_type_ids = np.array(val_token_type_ids)
    val_start_positions = np.array(val_start_positions)
    val_end_positions = np.array(val_end_positions)
    # 创建DataLoader
    train_dataset = TensorDataset(torch.tensor(train_input_ids), torch.tensor(train_attention_masks), torch.tensor(train_token_type_ids), torch.tensor(train_start_positions), torch.tensor(train_end_positions))
    val_dataset = TensorDataset(torch.tensor(val_input_ids), torch.tensor(val_attention_masks), torch.tensor(val_token_type_ids), torch.tensor(val_start_positions), torch.tensor(val_end_positions))
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)  # 验证集不随机采样，顺序读取
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size)

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


    eval_training(num_epochs)

    # 保存微调后的模型
    model.save_pretrained("data/SQuAD/SQuAD_finetuned_bert")