# 7.07 GNN 训练脚本 
#  
# 特别谢鸣 王赫萌 先生，提供了此代码的基准
# 在此基础上修改为我所需要的版本：fp32 | AMP版
# 
# python train_gnn.py -net GCN -dataset cora -precision fp32
# python train_gnn.py -net GCN -dataset cora -precision amp
import argparse
import torch
import torch.nn as nn
from utils import load_data, torch_cuda_active
import timeit
from torch.cuda.amp import GradScaler, autocast

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def emp_train(g, features, labels, masks, model, args):
    model.train()
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)
    
    for epoch in range(args.epoch):
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    acc = evaluate(g, features, labels, val_mask, model)
    print("EMP  | Loss {:.4f} | Accuracy {:.4f} ".format(loss.item(), acc))

def fp32_train(g, features, labels, masks, model, args):
    # define train/val samples, loss function and optimizer
    model.train()
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)
    
    # training loop
    for epoch in range(args.epoch):
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    acc = evaluate(g, features, labels, val_mask, model)
    print("FP32 | Loss {:.4f} | Accuracy {:.4f} ".format(loss.item(), acc))
    
def amp_train(g, features, labels, masks, model, args, amp_dtype):
    # define train/val samples, loss function and optimizer
    model.train()
    amp_enabled = amp_dtype in (torch.float16, torch.float32)
    scaler = GradScaler(enabled=True)
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    # training loop
    for epoch in range(args.epoch):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        scaler.scale(loss).backward()  
        scaler.step(optimizer)  
        scaler.update()     

    acc = evaluate(g, features, labels, val_mask, model)
    print("AMP  | Loss {:.4f} | Accuracy {:.4f} ".format(loss.item(), acc))

if __name__ == "__main__":
    torch.manual_seed(0) 
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, default='GCN', help="which GNN: GCN | GAT")
    parser.add_argument("-dataset", type=str, default='reddit', help="the input dataset")
    parser.add_argument("-epoch", type=int, default=100, help="the number of training epochs")
    parser.add_argument("-n_hidden", type=int, default=128, help="the number of hidden units")
    parser.add_argument("-n_layers", type=int, default=2, help="the number of GCN layers")
    parser.add_argument("-sub_rate", type=float, default=None, help="the number representing the ratio of subgraph")
    parser.add_argument('-precision', type=str, default='fp32', help='use which precision: fp32|amp')
    parser.add_argument("-repeat_times", type=int, default=10, help="the number repeat to train")
    parser.add_argument("-device", type=str, default='cuda', help="the training device")
    args = parser.parse_args()

    data = load_data(args)
    g = data[0]

    device = torch_cuda_active()
    g = g.int().to(device)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    n_class = g.ndata['label'].max().item() + 1

    
    in_size = features.shape[1]
    out_size = n_class
    
    train_start_time = timeit.default_timer()
    for i in range(args.repeat_times):   
        if args.precision != 'emp':
            # create model
            if args.net == 'GCN':
                from models.gnn import GCN_flatten
                model = GCN_flatten(in_size, args.n_hidden, out_size).to(device)
            elif args.net == 'GAT':
                from models.gnn import GAT_flatten
                model = GAT_flatten(in_size, args.n_hidden, out_size, heads=[8, 1]).to(device)

            # model training
            if args.precision == 'fp32':
                fp32_train(g, features, labels, masks, model, args)
            elif args.precision == 'amp':
                amp_dtype = torch.float16 # 在论文中要写明，GNN实验时 用的是float16而非bfloat16
                amp_train(g, features, labels, masks, model, args, amp_dtype)
                
        elif args.precision == 'emp':
            if args.net == 'GCN':
                model_layer_num = 5
                policy_precision_string = '00000'
                from models.gnn import GCN_flatten_emp
                model = GCN_flatten_emp(in_size, args.n_hidden, out_size,  policy_precision_string=policy_precision_string).to(device)
            elif args.net == 'GAT':
                model_layer_num = 4
                policy_precision_string = '0'*model_layer_num
                from models.gnn import GAT_flatten_emp
                model = GAT_flatten_emp(in_size, args.n_hidden, out_size, heads=[8, 1],  policy_precision_string=policy_precision_string).to(device)
            
            emp_train(g, features, labels, masks, model, args)
    
    torch.cuda.synchronize()
    train_end_time = timeit.default_timer()
    
    
    # print_model_summary(model, (in_size,))
    

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
        

    print('\n\nTraining summary', '-'*50,'\n{} {} {} epoch, \tPrecision Policy: {}'.format(args.net, args.dataset, args.epoch, args.precision))
    print('Each epoch average cost time: {:.6f} sec, \tFinal Accuracy: {:.4f}'.format((train_end_time - train_start_time) / (args.epoch * args.repeat_times), acc))
    print('Max GPU memory: {:.2f} MB'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)))
    
    if args.epoch <= 5: 
        summary_info_txt_filename = 'Log_Performance_GPU_memory.txt'
    else:
        summary_info_txt_filename = 'Log_Accuracy.txt'
        
    with open(summary_info_txt_filename, 'a') as f: 
        print('{} {} {} epoch, \tPrecision Policy: {}'.format(args.net, args.dataset, args.epoch, args.precision), file=f)
        print('Each epoch average cost time: {:.2f} sec, \tFinal Accuracy: {:.4f}'.format((train_end_time - train_start_time) / (args.epoch * args.repeat_times), acc), file=f)
        print('Max GPU memory: {:.2f} MB\n'.format(torch.cuda.max_memory_allocated() / (1024 ** 2)), file=f)
