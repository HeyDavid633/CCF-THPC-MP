import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import FlickrDataset, RedditDataset, YelpDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from sklearn.preprocessing import StandardScaler
import numpy as np

dgl_dataset_path = '/workspace/CCF-THPC-MP/data/dgl'

def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    if name == 'ogbn-mag':
        g = dgl.to_homogeneous(g)
        n_node = g.num_nodes()
        num_train = int(n_node * 0.9)
        num_val = int(n_node * 0.05)
        g.ndata['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
        g.ndata['train_mask'][:num_train] = True
        g.ndata['val_mask'][num_train:num_train + num_val] = True
        g.ndata['test_mask'][num_train + num_val:] = True
        label = torch.randint(0, 10, (n_node, ))
        g.ndata['label'] = label
        g.ndata['feat'] = torch.rand(n_node, 128)
        return g

    n_node = g.num_nodes()
    node_data = g.ndata

    if name == 'ogbn-proteins':
        label = torch.randint(0, 10, (n_node, ))
        node_data['label'] = label
        g.ndata['feat'] = torch.rand(n_node, 128)
    else:
        node_data['label'] = label.view(-1).long()

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True

    return g


def sub_save_graph(g, path):
    dgl.save_graphs(path, g)


def load_sub_graph(path):
    sub_g = dgl.load_graphs(path)[0][0]
    return sub_g


def sample_sub_graph(g, ratio):
    n = g.num_nodes()
    nodes = np.arange(int(n * ratio))
    sub_g = dgl.node_subgraph(graph=g, nodes=nodes)
    return sub_g


def load_data(args):
    if args.dataset == 'reddit':
        data = RedditDataset(raw_dir=dgl_dataset_path)
        g = data[0]
    elif args.dataset == 'ogbn-arxiv':
        g = load_ogb_dataset('ogbn-arxiv')
    elif args.dataset == 'ogbn-proteins':
        g = load_ogb_dataset('ogbn-proteins')
    elif args.dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif args.dataset == 'ogbn-mag':
        g = load_ogb_dataset('ogbn-mag')
    elif args.dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif args.dataset == 'yelp':
        data = YelpDataset()
        g = data[0]
        g.ndata['label'] = g.ndata['label'].float()
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset == 'flickr':
        data = FlickrDataset()
        g = data[0]
        g.ndata['label'] = g.ndata['label'].long()
        g.ndata['train_mask'] = g.ndata['train_mask'].bool()
        g.ndata['val_mask'] = g.ndata['val_mask'].bool()
        g.ndata['test_mask'] = g.ndata['test_mask'].bool()
        feats = g.ndata['feat']
        scaler = StandardScaler()
        scaler.fit(feats[g.ndata['train_mask']])
        feats = scaler.transform(feats)
        g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    elif args.dataset == "cora":
        data = CoraGraphDataset()
        g = data[0]
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
        g = data[0]
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
        g = data[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()

    if args.sub_rate != None :
        print("Subsampling the graph with rate", args.sub_rate)
        g = sample_sub_graph(g, args.sub_rate)

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    return g, n_feat, n_class


import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        default='reddit',
                        help="the input dataset")
    parser.add_argument("--n-epochs",
                        "--n_epochs",
                        type=int,
                        default=100,
                        help="the number of training epochs")
    parser.add_argument("--n-hidden",
                        "--n_hidden",
                        type=int,
                        default=128,
                        help="the number of hidden units")
    parser.add_argument("--n-layers",
                        "--n_layers",
                        type=int,
                        default=2,
                        help="the number of GCN layers")
    parser.add_argument("--sub-rate",
                        "--sub_rate",
                        type=float,
                        default=None,
                        help="the number representing the ratio of subgraph")
    parser.add_argument("--data-type",
                        "--data_type",
                        type=str,
                        default='bfloat16',
                        help="the train data type")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="the training device")


    return parser.parse_args()
