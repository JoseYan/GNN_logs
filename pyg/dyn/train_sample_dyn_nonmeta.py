#!/usr/bin/python3
from __future__ import division
import sys
import os
import subprocess
import time
import random
import argparse
import re
import logging
import os.path as osp
from sys import stdout
from urllib.parse import uses_params

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T

from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.data import GraphSAINTRandomWalkSampler,GraphSAINTNodeSampler,GraphSAINTEdgeSampler
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Flickr, Reddit2, Reddit,PPI

torch.manual_seed(0)
import random 
random.seed(0)
np.random.seed(0) 


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def load_data(name):
    if name == 'flickr':
        path = '/scratch/general/nfs1/u1320844/dataset/flickr'
        dataset = Flickr(path)
        data = dataset[0]
    elif name == 'reddit':
        path = '/scratch/general/nfs1/u1320844/dataset/reddit'
        dataset = Reddit(path)
        data = dataset[0]
    elif name == 'ppi':
        path = '/scratch/general/nfs1/u1320844/dataset'
        dataset = PPI(path)
        data = dataset[0]
    elif name == 'ogbn-arxiv':
        path = '/scratch/general/nfs1/u1320844/dataset'
        dataset = PygNodePropPredDataset(name=name,root=path)
        data = dataset[0]
        data.y = data.y.squeeze(dim=1)
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index)
        split_idx = dataset.get_idx_split()
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            if key=='valid':
                key = 'val'
            data[f'{key}_mask'] = mask
    elif name == 'ogbn-products':
        path = '/scratch/general/nfs1/u1320844/dataset'
        dataset = PygNodePropPredDataset(name=name, root=path)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.y = data.y.squeeze(dim=1)
        ## Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            if key=='valid':
                key = 'val'
            data[f'{key}_mask'] = mask
    elif name == 'ogbn-mag':
        path = '/scratch/general/nfs1/u1320844/dataset'
        dataset = PygNodePropPredDataset(name=name, root=path)
        rel_data = dataset[0]
        # only train with paper <-> paper relations.
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']['paper']
        val_idx = split_idx['valid']['paper']
        test_idx = split_idx['test']['paper']
        data.y = data.y.squeeze(dim=1)
        ## Convert split indices to boolean masks and add them to `data`.
        #for key, idx in split_idx.items():
        #    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        #    mask[idx] = True
        #    if key=='valid':
        #        key = 'val'
        #    data[f'{key}_mask'] = mask
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True
        data['train_mask'] = train_mask
        data['val_mask'] = val_mask
        data['test_mask'] = test_mask

    return data, dataset.num_classes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 128, cached=False)
        self.conv2 = GCNConv(128, int(num_classes), cached=False)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.dropout = args["dropout"]

    def forward(self, data, use_sparse=False):
        x = self.get_emb(data, use_sparse)
        return F.log_softmax(x, dim=1)

    def get_emb(self, data, use_sparse):
        if use_sparse:
            x, edge_index = data.x.float(), data.adj_t
        else:
            x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)#, return_attention_weights=True)
        return x

def train():
    model.train()
    total_loss = total_examples = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].long(), reduction='none')
        loss.mean().backward()
        optimizer.step()
        total_loss += loss.mean().item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples

@torch.no_grad()
def test_full(data):
    model.eval()
    data = T.ToSparseTensor()(data).to(device)
    logits, accs = model(data,True), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        _, pred = logits[mask].max(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

@torch.no_grad()
def test(loader):
    model.eval()
    for data in loader:
        data = data.to(device)
        logits, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            _, pred = logits[mask].max(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

    return accs

# Sample command
# -------------------------------------------------------------------
# python meta_gnn.py            --input /path/to/raw_files
#                               --name /name/of/dataset
#                               --output /path/to/output_folder
# -------------------------------------------------------------------

# Setup logger
#-----------------------

logger = logging.getLogger('GNN GraphSAINT')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHeader = logging.StreamHandler()
consoleHeader.setFormatter(formatter)
logger.addHandler(consoleHeader)

start_time = time.time()

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True, help="path to the input files")
ap.add_argument("-n", "--name", required=True, help="name of the dataset")
ap.add_argument("-o", "--output", required=True, help="output directory")
ap.add_argument("-l", "--loader", type=str, default='node') # sampler: node/edge/rw
ap.add_argument("--walk_len", type=int, default=3) # for rw sampler
ap.add_argument("-b", "--batch_size", type=int, default=20000) # node_budget, edge_budget and num_roots
ap.add_argument("-s", "--subgs", type=int, default=30) # number of subgraphs
ap.add_argument("-e", "--epochs", type=int, default=200) # number of epochs 
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--dropout", type=float, default=0)
ap.add_argument("--gpu", type=int, default=0, help="GPU index")
ap.add_argument("--load", action='store_true') # load processed graph from .pt file

args = vars(ap.parse_args())

input_dir = args["input"]
data_name = args["name"]
output_dir = args["output"]
loader_type = args["loader"]
bs = args["batch_size"]
n_subgs = args["subgs"]
load_from_disk = args["load"]
walk_len = args["walk_len"]
lr = args["lr"]
dr = args["dropout"]
gpu = args["gpu"]
epochs = args["epochs"]
print(args)

# Setup output path for log file
#---------------------------------------------------

fileHandler = logging.FileHandler(output_dir+"/"+"{}_gcn_{}_{}_{}_{}_{}_{}.log".format(data_name, loader_type, bs, n_subgs, lr, dr,walk_len))
#fileHandler = logging.FileHandler(output_dir+"/"+"metagnn_overlap_gcn_saint.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info("Input arguments:")
logger.info("Input dir: "+input_dir)
logger.info("Dataset: "+data_name)


data, num_classes = load_data(data_name)
num_features = len(data.x[0]) 

print(data)

logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

logger.info(args)

if loader_type == 'c':
    cluster_data = ClusterData(data, num_parts=n_subgs, recursive=False, save_dir=dataset.processed_dir)
    loader = ClusterLoader(cluster_data, batch_size=bs, shuffle=False, num_workers=5)
elif loader_type == 'node':
    loader = GraphSAINTNodeSampler(data,
                                    batch_size=bs,
                                    num_steps=n_subgs,
                                    sample_coverage=0,
                                    save_dir='overlap_node_subgs/')
elif loader_type == 'edge':
    loader = GraphSAINTEdgeSampler(data,
                                    batch_size=bs,
                                    num_steps=n_subgs,
                                    sample_coverage=0,
                                    save_dir='overlap_edge_subgs/')
elif loader_type == 'rw':
    loader = GraphSAINTRandomWalkSampler(data,
                                         batch_size=bs,
                                         walk_length=walk_len,
                                         num_steps=n_subgs,
                                         sample_coverage=0,
                                         save_dir='overlap_rw_subgs/')

device = torch.device(f'cuda:{gpu}')
logger.info("Running GNN on: "+str(device))
model = Net().to(device)
print(model)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=lr)

logger.info("Training model")
best_val_acc = test_acc = 0
total_time =0

best_test_acc = 0
epsilon = 0.0025 
best_epoch = 0

for epoch in range(800):
    ep_st = time.time()
    train()
    total_time += time.time()-ep_st
    train_acc, val_acc, tmp_test_acc = test_full(data)
    if tmp_test_acc > best_test_acc:
        best_test_acc = tmp_test_acc
        best_epoch = epoch
    else:
        if best_test_acc - tmp_test_acc <= epsilon and best_epoch <= 0.5 * epoch and epoch>40:
            # converged
            break

    #if val_acc > best_val_acc:
    #    best_val_acc = val_acc
    #    test_acc = tmp_test_acc
    log = 'Dataset:{}, Epoch: {:03d}, Time:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logger.info(log.format(data_name, epoch, total_time, train_acc, val_acc, tmp_test_acc))

elapsed_time = time.time() - start_time
# Print elapsed time for the process
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")
logger.info(f"Best acc: {best_test_acc:.4f}")


################################################################################################
# print("Embedding after training for node 0")
# data = data.to(device)
# new_emb, new_weights = model.get_emb(data)
# new_emb_arr = new_emb.detach().to("cpu").numpy()
# new_weights_arr = new_weights[1].detach().to("cpu").numpy()
# np.save(osp.join(input_dir, data_name, 'raw', 'learned_emb.npy'), new_emb_arr)
# np.save(osp.join(input_dir, data_name, 'raw', 'learned_weights.npy'), new_weights_arr)

#Print GCN model output
# output(output_dir, input_dir, data_name)

