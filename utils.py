""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
import numpy as np
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import FlickrDataset, RedditDataset, YelpDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


dgl_dataset_path = '/workspace/CCF-THPC-MP/data/dgl'
cifar100_dataset_path = '/workspace/CCF-THPC-MP/data/cifar-100-python'
imagenet_dataset_path = '/workspace/CCF-THPC-MP/data/imagenet'


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root=cifar100_dataset_path, train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root=cifar100_dataset_path, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def load_ImageNet(ImageNet_PATH = imagenet_dataset_path, batch_size=64, workers=3, pin_memory=True): 
    
    traindir = os.path.join(ImageNet_PATH, 'train')
    valdir   = os.path.join(ImageNet_PATH, 'val')
    print('traindir = ',traindir)
    print('valdir = ',valdir)
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ',len(train_dataset))
    print('val_dataset   = ',len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_dataset, val_dataset




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
        data = YelpDataset(raw_dir=dgl_dataset_path)
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
        data = FlickrDataset(raw_dir=dgl_dataset_path)
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
        data = CoraGraphDataset(raw_dir=dgl_dataset_path)
        g = data[0]
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(raw_dir=dgl_dataset_path)
        g = data[0]
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(raw_dir=dgl_dataset_path)
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



def torch_cuda_active():
    if torch.cuda.is_available():
        print('PyTorch version\t:', torch.__version__)
        print('CUDA version\t:', torch.version.cuda)
        print('GPU\t\t:', torch.cuda.get_device_name(), '\n')
        return torch.device('cuda')
    else:
        print('CUDA is not available!')
        return torch.device('cpu')
    
def policy_string_analyze(policy_precision_string):
    policy_precision = []
    for char in policy_precision_string:
        if char == '0':
            policy_precision.append(torch.float16)
        elif char == '1':
            policy_precision.append(torch.float32)
    return policy_precision