import argparse
import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data, torch_cuda_active
import timeit


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, args):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    train_start_time = timeit.default_timer()
    # training loop
    for epoch in range(args.epoch):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
            epoch, loss.item(), acc))
    
    torch.cuda.synchronize()
    print(f"[LOGS] {args.dataset},{args.data_type},{args.device},{timeit.default_timer() - train_start_time}")


if __name__ == "__main__":
    torch.manual_seed(0) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default='GCN', help="which GNN: GCN | GAT")
    parser.add_argument("--dataset", type=str, default='reddit', help="the input dataset")
    parser.add_argument("--epoch", type=int, default=10, help="the number of training epochs")
    parser.add_argument("--n_hidden", type=int, default=128, help="the number of hidden units")
    parser.add_argument("--n_layers", "--n_layers", type=int, default=2, help="the number of GCN layers")
    parser.add_argument("--sub_rate", type=float, default=None, help="the number representing the ratio of subgraph")
    parser.add_argument("--data_type",type=str, default='float32', help="the train data type")
    parser.add_argument("--device", type=str, default='cuda', help="the training device")
    args = parser.parse_args()

    data = load_data(args)
    g = data[0]

    device = torch_cuda_active()
    g = g.int().to(device)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    n_class = g.ndata['label'].max().item() + 1

    # create model
    in_size = features.shape[1]
    out_size = n_class
    
    if args.net == 'GCN':
        from models.gnn import GCN
        model = GCN(in_size, args.n_hidden, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.data_type == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, features, labels, masks, model, args)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
