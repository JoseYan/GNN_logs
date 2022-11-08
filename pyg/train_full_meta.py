#!/usr/bin/python3
from __future__ import division
import sys
import os
import subprocess
import csv
import operator
import time
import random
import argparse
import re
import logging
import os.path as osp
from sys import stdout

from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T

from Bio import SeqIO
from igraph import *
from collections import defaultdict
#from bidirectionalmap.bidirectionalmap import BidirectionalMap
from bidict import bidict

from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.data import GraphSAINTRandomWalkSampler,GraphSAINTNodeSampler,GraphSAINTEdgeSampler
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

torch.manual_seed(0)
import random 
random.seed(0)
np.random.seed(0) 

species_map = bidict()
#species_map = BidirectionalMap()

overlap_file_name = "overlap.gml"
species_list_file_name = "species.lst"

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def resize(l, newsize, filling=None):
    if newsize > len(l):
        l.extend([filling for x in range(len(l), newsize)])
    else:
        del l[newsize:]

def peek_line(f):
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line

tetra_list = []
def compute_tetra_list():
    for a in ['A', 'C', 'T', 'G']:
        for b in ['A', 'C', 'T', 'G']:
            for c in ['A', 'C', 'T', 'G']:
                for d in ['A', 'C', 'T', 'G']:
                    tetra_list.append(a+b+c+d)

def compute_tetra_freq(seq):
    tetra_cnt = []
    for tetra in tetra_list:
        tetra_cnt.append(seq.count(tetra))
    return tetra_cnt

def compute_gc_bias(seq):
    seqlist = list(seq)
    gc_cnt = seqlist.count('G') + seqlist.count('C')
    gc_frac = gc_cnt/len(seq)
    return gc_frac

def compute_contig_features(read_file, read_names):
    compute_tetra_list()
    gc_map = defaultdict(float) 
    tetra_freq_map = defaultdict(list)
    idx = 0
    for record in SeqIO.parse(read_file, 'fasta'):
        if record.name in read_names:
            gc_map[record.name] = compute_gc_bias(record.seq)
            tetra_freq_map[record.name] = compute_tetra_freq(record.seq)
        stdout.write("\r%d" % idx)
        stdout.flush()
        idx += 1
    return gc_map, tetra_freq_map

def read_features(gc_bias_f, tf_f):
    gc_map = pickle.load(open(gc_bias_f, 'rb'))
    tetra_freq_map = pickle.load(open(tf_f, 'rb'))
    return gc_map, tetra_freq_map

def write_features(file_name, gc_map, tetra_freq_map):
    gc_bias_f = file_name + '.gc'
    tf_f = file_name + '.tf'
    pickle.dump(gc_map, open(gc_bias_f, 'wb'))
    pickle.dump(tetra_freq_map, open(tf_f, 'wb'))
    
def read_or_compute_features(file_name, read_names):
    gc_bias_f = file_name + '.gc'
    tf_f = file_name + '.tf'
    if not os.path.exists(gc_bias_f) and not os.path.exists(tf_f):
        gc_bias, tf = compute_contig_features(file_name, read_names)
        write_features(file_name, gc_bias, tf)
    else:
        gc_bias, tf = read_features(gc_bias_f, tf_f)
    return gc_bias, tf


#def build_species_map(file_name):
#    overlap_graph = Graph()
#    overlap_graph = overlap_graph.Read_GML(file_name)
#    overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
#    
#    species = []
#    for v in overlap_graph.vs:
#        species.append(v['species'])
#
#    # prepare vertex labels
#    species_set = set(species)
#    idx = 0
#    for s in species_set:
#        species_map[s] = idx
#        idx += 1

def build_species_map(map_file, graph_file):
    species = []
    if not os.path.exists(map_file):
        overlap_graph = Graph()
        overlap_graph = overlap_graph.Read_GML(graph_file)
        overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
        species = []
        for v in overlap_graph.vs:
            species.append(v['species'])
        pickle.dump(species, open(map_file, 'wb'))
    else:
        species = pickle.load(open(map_file, 'rb'))
    
    # prepare vertex labels
    species_set = set(species)
    idx = 0
    for s in species_set:
        species_map[s] = idx
        idx += 1

def load_stored_graph(name):
    prefix = f'/scratch/general/nfs1/u1320844/dataset/asplos/{name}_subgs/'
    adj_full = torch.load(prefix+'adj_full.pt')
    x_full = torch.load(prefix+'x_full.pt')
    y_full = torch.load(prefix+'y_full.pt')
    print(f'{name}, {len(adj_full[0])}, {len(x_full)}, {len(adj_full[0])/len(x_full)}')
    train_mask_full = torch.load(prefix+'train_mask_full.pt')
    val_mask_full = torch.load(prefix+'val_mask_full.pt')
    test_mask_full = torch.load(prefix+'test_mask_full.pt')
    data = Data()
    data.edge_index = adj_full
    data.x = x_full
    data.y = y_full
    #data.x.requires_grad = True
    data.train_mask = train_mask_full
    data.val_mask = val_mask_full
    data.test_mask = test_mask_full
    return data

class Metagenomic(InMemoryDataset):
    r""" Assembly graph built over raw metagenomic data using spades.
        Nodes represent contigs and edges represent link between them.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"bacteria-10"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(Metagenomic, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['overlap.gml', 'reads.fa', 'species_all.graphml', 'species_training.graphml']
        # return ['minimap2.graphml', 'sampled.fq', 'species_all.graphml', 'species_training.graphml']

    @property
    def processed_file_names(self):
        return ['pyg_meta_graph.pt']

    def download(self):
        pass

    def process(self):
        overlap_graph_file = osp.join(self.raw_dir, self.raw_file_names[0])
        read_file = osp.join(self.raw_dir, self.raw_file_names[1])
        all_file = osp.join(self.raw_dir, self.raw_file_names[2])
        training_file = osp.join(self.raw_dir, self.raw_file_names[3])
        # Read assembly graph and node features from the file into arrays

        overlap_graph = Graph()
        overlap_graph = overlap_graph.Read_GML(overlap_graph_file)
        # overlap_graph = overlap_graph.clusters().subgraph(1)

        source_nodes = []
        dest_nodes = []
        # Add edges to the graph
        overlap_graph.simplify(multiple=True, loops=True, combine_edges=None)
        # overlap_graph.write_graphml(all_file)

        # prepare edge list
        for e in overlap_graph.get_edgelist():
            source_nodes.append(e[0])
            dest_nodes.append(e[1])

        node_count = overlap_graph.vcount()
        print("Nodes: " + str(overlap_graph.vcount()))
        print("Edges: " + str(overlap_graph.ecount()))
        clusters = overlap_graph.clusters()
        print("Clusters: " + str(len(clusters)))

        # get all vertex names
        vertex_names = []
        vertexes = overlap_graph.vs
        for v in overlap_graph.vs:
            vertex_names.append(v['name'])
        gc_map, tetra_freq_map = read_or_compute_features(read_file, vertex_names)
        # tetra_freq_map = tetra_freq_map[:,:64]

        # prepare node features
        node_gc = []
        node_tfq = []
        for v in overlap_graph.vs:
            node_gc.append(gc_map[v['name']])
            node_tfq.append(tetra_freq_map[v['name']])

        # prepare vertex labels
        node_labels = []
        for v in overlap_graph.vs:
            node_labels.append(species_map[v['species']])
        
        # prepare torch objects
        x = torch.tensor(node_tfq, dtype=torch.float)
        g = torch.tensor(node_gc, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.float)
        n = torch.tensor(list(range(0, node_count)), dtype=torch.int)
        edge_index = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)

        # prepare train/validate/test vectors
        # train_size = int(node_count/3)
        # val_size = train_size
        # train_index = torch.arange(train_size)
        # val_index = torch.arange(train_size, train_size+val_size)
        # test_index = torch.arange(train_size+val_size, node_count)
        # train_mask = index_to_mask(train_index, size=node_count)
        # val_mask = index_to_mask(val_index, size=node_count)
        # test_mask = index_to_mask(test_index, size=node_count)
        
        train_size = int(node_count/3)
        val_size = int(node_count/3)
        
        all_indexes = [i for i in range(node_count)]
        random.shuffle(all_indexes)
        train_index = all_indexes[0:train_size]
        val_index = all_indexes[train_size:train_size+val_size]
        test_index = all_indexes[train_size+val_size:]
    
        train_mask = index_to_mask(train_index, size=node_count)
        val_mask = index_to_mask(val_index, size=node_count)
        test_mask = index_to_mask(test_index, size=node_count)

        training_graph = overlap_graph
        vertex_set = training_graph.vs
        for i in range(node_count):
          if test_mask[i]:
            vertex_set[i]['species'] = 'Unknown'
        training_graph.write_graphml(training_file)
        learned_graph = training_graph

        data = Data(x=x[:,0:32], edge_index=edge_index, y=y, g=g, n=n)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data_list = []
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 128, cached=False)
        self.conv2 = GCNConv(128, int(num_classes), cached=False)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.dropout = args["dropout"]

    def forward(self, data):
        x = self.get_emb(data)
        return F.log_softmax(x, dim=1)

    def get_emb(self, data):
        #x, edge_index = data.x.float(), data.adj_t
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)#, return_attention_weights=True)
        return x

def train():
    model.train()
    total_loss = total_examples = 0
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
    #data = data.to(device)
    logits, accs = model(data), []
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

@torch.no_grad()
def output(output_dir, input_dir, data_name):
    overlap_graph_file = input_dir + '/' + data_name + '/raw/' + overlap_file_name
    for data in loader:
        data = data.to(device)
        _, preds = model(data).max(dim=1)
        acc = preds.eq(data.y).sum().item() / len(data.y)
        print(acc)
        learned_graph = Graph()
        learned_graph = learned_graph.Read_GML(overlap_graph_file)
        rev_species_map = species_map.inverse
        vertex_set = learned_graph.vs
        miss_pred_vertices = []
        print(data)
        perm = data.n.tolist()
        orgs = data.y.tolist()
        preds = preds.tolist()
        train = data.train_mask.tolist()
        # annotate graph
        for idx,org,pred,t in zip(perm,orgs,preds,train):
            if pred == org:
                vertex_set[idx]['pred'] = 'Correct'
            else:
                vertex_set[idx]['pred'] = 'Wrong'
            if t == 1:
                vertex_set[idx]['train'] = 'True'
                vertex_set[idx]['species'] = rev_species_map[org] 
            else:
                vertex_set[idx]['train'] = 'False'
                vertex_set[idx]['species'] = rev_species_map[pred] 
        # learned_file = output_dir + '/species_learned.graphml'
        # learned_graph.write_graphml(learned_file)
       
        t_idx_list = []
        for s in species_map:
            for v in vertex_set:
                if v['species'] == s:
                    t_idx_list.append(v.index)
                    break

        # print a subgraph
        edge_set = set()
        for idx in t_idx_list:
            bfsiter = learned_graph.bfsiter(vertex_set[idx], OUT, True)
            for v in bfsiter:
                if v[1] < 3: 
                    if v[1] > 0:
                        edge_set.add(learned_graph.get_eid(v[2].index, v[0].index))
                        # subvertex_set.add(v[2].index)
                        # subvertex_set.add(v[0].index)

        subedge_list = list(edge_set)
        subgraph = learned_graph.subgraph_edges(subedge_list)
        print(subgraph.vcount())
        print(subgraph.ecount())
        # subvertex_list = list(subvertex_set)
        # subgraph = learned_graph.subgraph(subvertex_list)
        subgraph_file = output_dir + '/species_subgraph.graphml'
        subgraph.write_graphml(subgraph_file)

# Sample command
# -------------------------------------------------------------------
# python meta_gnn.py            --input /path/to/raw_files
#                               --name /name/of/dataset
#                               --output /path/to/output_folder
# -------------------------------------------------------------------

# Setup logger
#-----------------------

logger = logging.getLogger('MetaGNN 1.0')
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

fileHandler = logging.FileHandler(output_dir+"/"+"{}_gcn_full_{}_{}.log".format(data_name, lr, dr))
#fileHandler = logging.FileHandler(output_dir+"/"+"metagnn_overlap_gcn_saint.log")
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

logger.info("Welcome to MetaGNN: Metagenomic reads classification using GNN.")
logger.info("This version of MetaGNN makes use of the overlap graph produced by Minimap2.")

logger.info("Input arguments:")
logger.info("Input dir: "+input_dir)
logger.info("Dataset: "+data_name)

logger.info("MetaGNN started")

logger.info("Constructing the overlap graph and node feature vectors")

if load_from_disk:
    data = load_stored_graph(data_name)
    #data = T.ToSparseTensor()(data)
    num_features = len(data.x[0]) 
    if data_name == 'airways':
        num_classes = 25 
    elif data_name == 'arctic25':
        num_classes = 33 
    elif data_name == 'oral':
        num_classes = 32
else:
    build_species_map(osp.join(input_dir, data_name, 'raw', species_list_file_name), osp.join(input_dir, data_name, 'raw', overlap_file_name))
    dataset = Metagenomic(root=input_dir, name=data_name)#, transform=T.ToSparseTensor())
    data = dataset[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes

print(data)

logger.info("Graph construction done!")
elapsed_time = time.time() - start_time
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")

logger.info(args)

#if loader_type == 'c':
#    cluster_data = ClusterData(data, num_parts=n_subgs, recursive=False, save_dir=dataset.processed_dir)
#    loader = ClusterLoader(cluster_data, batch_size=bs, shuffle=False, num_workers=5)
#elif loader_type == 'node':
#    loader = GraphSAINTNodeSampler(data,
#                                    batch_size=bs,
#                                    num_steps=n_subgs,
#                                    sample_coverage=0,
#                                    save_dir='overlap_node_subgs/')
#elif loader_type == 'edge':
#    loader = GraphSAINTEdgeSampler(data,
#                                    batch_size=bs,
#                                    num_steps=n_subgs,
#                                    sample_coverage=0,
#                                    save_dir='overlap_edge_subgs/')
#elif loader_type == 'rw':
#    loader = GraphSAINTRandomWalkSampler(data,
#                                         batch_size=bs,
#                                         walk_length=walk_len,
#                                         num_steps=n_subgs,
#                                         sample_coverage=0,
#                                         save_dir='overlap_rw_subgs/')

device = torch.device(f'cuda:{gpu}')
logger.info("Running GNN on: "+str(device))
model = Net().to(device)
print(model)
data = data.to(device)

optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=lr)

logger.info("Training model")
best_val_acc = test_acc = 0
total_time =0
for epoch in range(epochs):
    ep_st = time.time()
    train()
    total_time += time.time()-ep_st
    train_acc, val_acc, tmp_test_acc = test_full(data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Dataset:{}, Epoch: {:03d}, Time:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    logger.info(log.format(data_name, epoch, total_time, train_acc, best_val_acc, tmp_test_acc))

elapsed_time = time.time() - start_time
# Print elapsed time for the process
logger.info("Elapsed time: "+str(elapsed_time)+" seconds")
logger.info(f"Best acc: {test_acc:.4f}")


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

