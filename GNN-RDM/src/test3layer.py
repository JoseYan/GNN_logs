import os
import os.path as osp
import argparse

import math

import torch
import torch.distributed as dist

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from reddit import Reddit
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, to_dense_adj, dense_to_sparse, to_scipy_sparse_matrix
import torch_geometric.transforms as T

import torch.multiprocessing as mp

from torch.multiprocessing import Manager, Process

from torch.nn import Parameter
import torch.nn.functional as F

from torch_scatter import scatter_add
import torch_sparse

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

import socket
import statistics
import time
import numpy as np

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.
    #num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
         edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                  device=edge_index.device)
    if add_self_loops:
         edge_index, tmp_edge_weight = add_remaining_self_loops(
             edge_index, edge_weight, fill_value, num_nodes)
         assert tmp_edge_weight is not None
         edge_weight = tmp_edge_weight
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    #print('normalized adj val:', deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col])
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
# comp_time = 0.0
# comm_time = 0.0
# scomp_time = 0.0
# dcomp_time = 0.0
# bcast_comm_time = 0.0
# bcast_words = 0
# op1_comm_time = 0.0
# op2_comm_time = 0.0
total_time = dict()
comp_time = dict()
comm_time = dict()
scomp_time = dict()
dcomp_time = dict()
bcast_comm_time = dict()
barrier_time = dict()
barrier_subset_time = dict()
op1_comm_time = dict()
op2_comm_time = dict()

epochs = 0
graphname = ""
mid_layer = 0
timing = True
normalization = False
activations = False
accuracy = False
device = None
acc_per_rank = 0
run_count = 0
run = 0
download = False
ht = True
mmorder = "dsds"

def start_time(group, rank, subset=False, src=None):
    global barrier_time
    global barrier_subset_time

    if not timing:
        return 0.0
    
    barrier_tstart = time.time()

    dist.barrier(group)
    barrier_tstop = time.time()
    barrier_time[run][rank] += barrier_tstop - barrier_tstart
    #if rank == 0:        print(f'barrier_time {barrier_tstop - barrier_tstart:0.6f}')
    if subset:
        barrier_subset_time[run][rank] += barrier_tstop - barrier_tstart

    tstart = 0.0
    tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    if not timing:
        return 0.0
    dist.barrier(group)
    devid = rank_to_devid(rank, acc_per_rank)
    device = torch.device('cuda:{}'.format(devid))
    # torch.cuda.synchronize(device=device)

    tstop = time.time()
    return tstop - tstart

def normalize(adj_matrix):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    d = torch.sum(adj_matrix, dim=1)
    d = torch.rsqrt(d)
    d = torch.diag(d)
    return torch.mm(d, torch.mm(adj_matrix, d))

def block_row(adj_matrix, am_partitions, inputs, weight, rank, size):
    n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    # n_per_proc = int(adj_matrix.size(1) / size)
    # am_partitions = list(torch.split(adj_matrix, n_per_proc, dim=1))

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.zeros(inputs.size())

    part_id = rank % size

    z_loc += torch.mm(am_partitions[part_id].t(), inputs) 

    for i in range(1, size):
        part_id = (rank + i) % size

        inputs_recv = torch.zeros(am_partitions[part_id].size(0), inputs.size(1))

        src = (rank + 1) % size
        dst = rank - 1
        if dst < 0:
            dst = size - 1

        if rank == 0:
            dist.send(tensor=inputs, dst=dst)
            dist.recv(tensor=inputs_recv, src=src)
        else:
            dist.recv(tensor=inputs_recv, src=src)
            dist.send(tensor=inputs, dst=dst)
        
        inputs = inputs_recv.clone()

        # z_loc += torch.mm(am_partitions[part_id], inputs) 
        z_loc += torch.mm(am_partitions[part_id].t(), inputs) 


    # z_loc = torch.mm(z_loc, weight)
    return z_loc

def outer_product(adj_matrix, grad_output, rank, size, group):
    global comm_time
    global comp_time
    global dcomp_time
    global op1_comm_time
    global run


    n_per_proc = math.ceil(float(adj_matrix.size(0)) / size)
    
    tstart_comp = start_time(group, rank)

    # A * G^l
    ag = torch.mm(adj_matrix, grad_output)

    dur = stop_time(group, rank, tstart_comp)
    comp_time[run][rank] += dur
    dcomp_time[run][rank] += dur

    tstart_comm = start_time(group, rank)

    # reduction on A * G^l low-rank matrices
    dist.all_reduce(ag, op=dist.ReduceOp.SUM, group=group)

    dur = stop_time(group, rank, tstart_comm)
    comm_time[run][rank] += dur
    op1_comm_time[run][rank] += dur

    # partition A * G^l by block rows and get block row for this process
    # TODO: this might not be space-efficient
    red_partitions = list(torch.split(ag, n_per_proc, dim=0))
    grad_input = red_partitions[rank]

    return grad_input

def outer_product2(inputs, ag, rank, size, group):
    global comm_time
    global comp_time
    global dcomp_time
    global op2_comm_time
    global run

    #if rank == 0 : print('weight_compute ',end='')
    tstart_comp = start_time(group, rank)
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)

    dur = stop_time(group, rank, tstart_comp)
    comp_time[run][rank] += dur
    dcomp_time[run][rank] += dur
    
    #if rank == 0 : print('weight_reduce ',end='')
    tstart_comm = start_time(group, rank)
    # reduction on grad_weight low-rank matrices
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=group)

    dur = stop_time(group, rank, tstart_comm)
    comm_time[run][rank] += dur
    op2_comm_time[run][rank] += dur

    return grad_weight

def transpose_input(inputs,rank,size,row_group,group,dim):
    #print('in transpose '+str(rank)+' at dim '+str(dim))
    global device
    global comm_time
    global comp_time
    global scomp_time
    global bcast_comm_time
    global run
    global ht
    global replication
    
    # if dim = 0 vertical tiling, dim = 1 horizontal tiling        
    col_count = size//replication    
    rep_id = rank//col_count
    ht = (dim == 1)    
    # size//replication must be a power of 2
    rank_mask =  col_count - 1 
    split_size = [ math.ceil(float(inputs.size(1-dim)) / col_count) ] * col_count
    split_size[col_count-1] = inputs.size(1-dim) - math.ceil(float(inputs.size(1-dim)) / col_count) * (col_count-1)
    if split_size[col_count-1] <= 0:
       split_sum = 0
       for i in range(col_count):
          split_size[i] = math.ceil((i+1)*float(inputs.size(1-dim)) / col_count) - split_sum
          split_sum += split_size[i]
    #print(split_size)
    #exit()
    #input_2d = torch.split(inputs, math.ceil(float(inputs.size(1-dim)) / col_count), dim=1-dim)    
    input_2d = torch.split(inputs, split_size, dim=1-dim)
    recv = [None]*col_count
    #print(dim_count)
    if dim == 0:
        for i in range(col_count):
            #print(i)
            try:
                dim_count[rep_id][1][i] = input_2d[i].size(1)
            except:
                print(f'Index error at redistribution at index {i} rep_id {rep_id}')
                #print(f'Col processes per row panel {col_count}')
                #print(f'Total number of cols {math.ceil(float(inputs.size(1-dim)))}')
                #print(f'Cols per process {math.ceil(float(inputs.size(1-dim)) / col_count)}')
                #print(dim_count[rep_id])
                #print(dim_count[rep_id][1])
                #print(input_2d)
                #print(col_count)
                #print(len(input_2d))
                exit()
    
    #print('\ndim is '+str(dim))
    tstart_comm = start_time(group, rank)     
    for i in range(col_count):
        p_i = i^rank        
        il = p_i & rank_mask
        #print(str(il)+' il '+str(rank)+' rank '+str(p_i)+' p_i and i'+str(i))
        if p_i == rank:
            #print('self copy at rank '+str(rank))
            recv[il] = input_2d[il]
            continue
        
        recv_row = (p_i if dim == 0 else rank) & rank_mask
        recv_col = (p_i if dim == 1 else rank) & rank_mask
        #print(recv_row)
        #print(recv_col)
        #print(str(dim_count)+' ' +str(rank)+' '+str(p_i) + ' ' +str(recv_row))
        #print(str(rank)+ " " +str([dim_count[rep_id][0][recv_row],dim_count[rep_id][1][recv_col]]))
        input_recv = torch.zeros(dim_count[rep_id][0][recv_row], dim_count[rep_id][1][recv_col], device=device)   
        #print(input_recv.size())
        if p_i < rank :
            dist.send(tensor=input_2d[il].contiguous(), dst=p_i)
            dist.recv(tensor=input_recv, src=p_i)
        else:
            dist.recv(tensor=input_recv, src=p_i)
            dist.send(tensor=input_2d[il].contiguous(), dst=p_i)
            
        #print(input_recv.size())
        
        recv[il] = input_recv
    #print('ht '+str(ht))
    #print('dim '+str(dim))
    #print('row_group '+str(row_group))
    
    inputs = torch.cat(recv,dim)    
    dur = stop_time(group, rank, tstart_comm)
    comm_time[run][rank] += dur
    bcast_comm_time[run][rank] += dur
    return inputs


def spmm_func(am_partitions, inputs, rank, size, group, row_group, col_group, horizontal_tiled):
    global device
    global comm_time
    global comp_time
    global scomp_time
    global bcast_comm_time
    global run
    global replication    
    
    #print('spmm at rank '+str(rank))
    col_count = size//replication
    rep_id = rank//col_count
    inputs_ = inputs.detach().contiguous() #torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)
    if horizontal_tiled:
        #if rank == 0: print('spmm_htv ',end='')
        inputs_ = transpose_input(inputs,rank,size,row_group,group,0)   
        #dist.barrier(row_group)
        #if rank == 0:           print('spmm row group')
    #else:
    #    dist.barrier(col_group)
        #if rank == 0:           print('spmm col group')
    #print(am_partitions[0].size())
    #print(inputs_.size())
    z_loc = torch.cuda.FloatTensor(am_partitions[rep_id].size(0), inputs_.size(1), device=device).fill_(0)
    rank_col = rank % col_count
    
    for i in range(replication):  
        # print('size of output '+ str(z_loc.size()))
        
        # Broadcast dense matrix
        tile_id = ( rank_col ) + i*col_count
        inputs_recv = 0
        if tile_id == rank:
            inputs_recv = inputs_
        else:
            row_recv = sum(dim_count[i][0])            
            # could cause cuda oom
            inputs_recv = torch.cuda.FloatTensor(row_recv, inputs_.size(1), device=device).fill_(0)
        #if rank == 0: print('spmm_bcast ',end='')                                
        tstart_comm = start_time(group, rank)        
        dist.broadcast(inputs_recv, src=tile_id, group=col_group)        
        dur = stop_time(group, rank, tstart_comm)
        comm_time[run][rank] += dur
        bcast_comm_time[run][rank] += dur                
        #if rank == 0: print('spmm_compute ',end='')
        tstart_comp = start_time(group, rank)
        #print('SpMM input '+str(inputs_recv.size())+" output "+str(z_loc.size()) + ' at rank '+str(rank))
        #print(am_partitions[i].size())
        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                            am_partitions[i].values(), am_partitions[i].size(0), 
                            am_partitions[i].size(1), inputs_recv, z_loc)

        dur = stop_time(group, rank, tstart_comp)
        comp_time[run][rank] += dur
        scomp_time[run][rank] += dur
    
    #dist.barrier(col_group)    
    return z_loc


def gemm_func(inputs, weight, rank, size, group, row_group, col_group, horizontal_tiled):
    global device
    global comm_time
    global comp_time
    global scomp_time
    global bcast_comm_time
    global run

    #dist.barrier(row_group)
    dist.barrier(group)
    #print('gemm at rank '+str(rank))
    col_count = size//replication
    rep_id = rank//col_count    
    inputs_recv = inputs 
    if rank == 0:    print('1->',end=' ')
    if not horizontal_tiled:
        if rank == 0: print('2->',end='')
        inputs_recv = transpose_input(inputs,rank,size,row_group,1)

    if rank == 0:    print('3->',end='')
    tstart_comp = start_time(group, rank)
    if rank == 0:    print('4->',end='')
    z = torch.mm(inputs_recv, weight)
    dur = stop_time(group, rank, tstart_comp)
    comp_time[run][rank] += dur
    dcomp_time[run][rank] += dur
    if rank == 0:        print('5->',end='')
    dist.barrier(group)
    #dist.barrier(row_group)
    if rank == 0:        print('6')
    return z

def dist_nll(outputs,labels,row_group,rank):
    loss = 0
    col_count = size // replication
    col_id = rank % col_count       
    l_start = sum[dim_count[0][1][:col_id]] # Sum the col_ids until current col
    l_end = sum[dim_count[0][1][:col_id+1]] # Sum the col_ids including current col
    for i in range(output.size(0)):
        if labels[i]>=l_start and labels[i] < l_end:
            loss -= outputs[labels[i]-l_start]
    # reduce loss
    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=row_group)
    loss /= output.size(0)
    return loss

def dist_log_softmax(z, rank, size, group):
    print('dist log softmax')
    max_row = z.max(dim=1).values
    
    # Communicate max_row between nodes and update it    
    dist.all_reduce(max_row, op=dist.ReduceOp.MAX, group=group)

    h_norm_row = (z - max_row[:, None])
    h_row_sum = h_norm_row.exp().sum(dim=1).log()
    h = h_norm_row - h_row_sum[:, None]    

    return h
    

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, order, last_layer, func):
        global comm_time
        global comp_time
        global dcomp_time
        global run        
        global ht
        global replication
        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        # adj_matrix = adj_matrix.to_dense()
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.am_partitions = am_partitions
        ctx.rank = rank
        ctx.size = size
        ctx.group = group
        ctx.func = func
        ctx.order = order
        col_count = size // replication
        row_id = rank // col_count
        col_id = rank % col_count 
        row_group = row_groups[row_id]
        col_group = col_groups[col_id]
        ctx.row_group = row_group
        ctx.col_group = col_group
        ctx.last = last_layer
        ctx.x1 = None
        ctx.input_h = None
        input_ht = ht 
        if order == "dd":
            print("ERROR: Suboptimal computation configuration. SpMM should be the first op in either forward or backward. Quitting...")
            exit()
        z = 0
        #if rank == 0: print(inputs.size(),end=', ')
        if order[0] == 'd':
            if order[1] == 's':
                if input_ht:
                    ctx.input_h = inputs
                else:
                    input_h = transpose_input(inputs,rank,size,row_group,group,1)
                    ctx.input_h = input_h
                    inputs = input_h
            if rank == 0: print('forw horizontal '+str(rank),end=' ->')
            z = gemm_func(inputs, weight, rank, size, group, row_group, col_group, ht)
            z = spmm_func(am_partitions, z, rank, size, group, row_group, col_group, ht)
            
                
        else:
            if rank == 0: print('forw vertical '+str(rank),end=' ->')
            # z = block_row(adj_matrix.t(), am_partitions, inputs, weight, rank, size)
            y1 = spmm_func( am_partitions, inputs, rank, size, group, row_group, col_group, ht)
            
            # Convert y1 to x1
            #if rank == 0:                print('intra_layer_v2h ',end='') 
            x1 = transpose_input(y1,rank,size,row_group,group,1)
            ctx.x1 = x1
            if order[1] == 's':
                if input_ht:
                    ctx.input_h = inputs
                else:
                    #if rank == 0:                        print('intralayer_extra_v2h ',end='')
                    input_h = transpose_input(inputs,rank,size,row_group,group,1)
                    ctx.input_h = input_h
            z = gemm_func(x1, weight, rank, size, group, row_group, col_group, ht)

        if last_layer and not ht:
            #if rank == 0: print('lastlayer_v2h ',end='')
            z = transpose_input(z,rank,size,row_group,group,1)
        z.requires_grad = True                
        ctx.z = z
        #ht = not ht

        
        #print('z '+str(z.size()))
        if activations:
            if func is F.log_softmax:
                if ht:
                    h = func(z, dim=1)
                else:
                    h = dist_log_softmax(z, rank, size, row_group)
            elif func is F.relu:
                h = func(z)
            else:
                h = z

            return h
        else:
            return z

    @staticmethod
    def backward(ctx, grad_output):
        global comm_time
        global comp_time
        global dcomp_time
        global run
        global ht
        global x1

        #if ctx.rank == 0: print('grad back '+str(grad_output.size()))
        inputs, weight, adj_matrix = ctx.saved_tensors
        am_partitions = ctx.am_partitions
        rank = ctx.rank
        size = ctx.size
        group = ctx.group
        order = ctx.order
        row_group = ctx.row_group
        col_group = ctx.col_group

        func = ctx.func
        z = ctx.z

        if activations:
            with torch.set_grad_enabled(True):
                if func is F.log_softmax:
                    if ht:
                        func_eval = func(z, dim=1)
                    else:
                        func_eval = dist_log_softmax(z, rank, size, row_group)
                elif func is F.relu:
                    func_eval = func(z)
                else:
                    func_eval = z

                sigmap = torch.autograd.grad(outputs=func_eval, inputs=z, grad_outputs=grad_output)[0]
                grad_output = sigmap
                #rint(grad_output.size())

        #print('grad output '+str(grad_output[0]) + ' rank ' + str(rank))
        grad_input=None
        grad_weight=None


        #if rank == 0: print(grad_output.size(),end=', ')
        if order[1] == 'd':
            #if rank == 0: print('back horizontal '+str(rank),end=' ->')
            #if rank == 0: print('params: '+str(rank),end=' ->')
            x2 = gemm_func(grad_output, weight.t(), rank, size, group, row_group, col_group, ht)
            #if rank==0:
            #    print('AAAAAA')
            grad_input = spmm_func(am_partitions, x2 , rank, size, group, row_group, col_group, ht)
            #if rank==0:
            #    print('done spmm')
            # Second backprop equation (reuses the A * H^(l-1) computation)
            x1 = ctx.x1
            #print(x1.size())
            #print(grad_output.size())
            grad_weight = outer_product2(x1.t(), grad_output, rank, size, group)

        else:
            if rank == 0: print('back vertical '+str(rank),end=' ->')
            #print(grad_output.size())
            ag = spmm_func(am_partitions, grad_output, rank, size, group, row_group, col_group, ht)            

            # Convert ag to horizontal tiling
            #if rank == 0: print('intralayer_v2h ',end='')
            ag = transpose_input(ag,rank,size,row_group,group,1)

            print('here')
            grad_input = gemm_func(ag, weight.t(), rank, size, group, row_group, col_group, ht)
            # Second backprop equation (reuses the A * G^l computation)
            # Convert tiling of input if it is vertically tiled
            #print(str(inputs.size(1))+" expected but got "+str(ag.size(1)))
            inputs_h = inputs
            if inputs.size(0) != ag.size(0):
                inputs_h = ctx.input_h
            ht = True            
            grad_weight = outer_product2(inputs_h.t(), ag, rank, size, group)
            #print(grad_weight.size())

        if inputs.size(1) != grad_input.size(1) :
            dim = 0 if inputs.size(1) < grad_input.size(1) else 1
            #if rank == 0: print('weightupdate_redist ',end='')
            grad_input = transpose_input(grad_input,rank,size,row_group,group,dim)

        # print('grad input '+str(grad_input[0]) + ' rank ' + str(rank))
        # print('grad weight '+str(grad_weight[0]) + ' rank ' + str(rank))
        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, order, data, rank, size, group, row_groups, col_groups, horizontal_tiled):
    global ht
    global run
    
    order1 = order[0] + order[3]
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, order1, False, F.relu)
    order2 = order[1] + order[2]
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, rank, size, group, row_groups, col_groups, order2, True, F.log_softmax)
   
    # print(outputs) 
    optimizer.zero_grad()
    # print(data.train_mask.bool().size())
    # print(data.y.size())
    rank_train_mask = data.train_mask.bool()
    datay_rank = data.y
    
    v_size = []
    for c in range(replication):
        v_size += dim_count[c][0]    
    if ht:
        rank_train_mask = torch.split(data.train_mask.bool(), v_size, dim=0)[rank]
        datay_rank = torch.split(data.y, v_size, dim=0)[rank]
    else:        
        print("ERROR:")
        exit()
        #print('dim '+str(0 if inputs.size(1) < grad_input.size(1) else 1))
        #print(str(inputs.size()) + ' but got ' + str(grad_input.size()))
        rep_id = rank // (size // replication)
        outputs = transpose_input(outputs,rank,size, row_groups[rep_id],group, 1)        
        rank_train_mask = torch.split(data.train_mask.bool(), v_size, dim=0)[rank]
        datay_rank = torch.split(data.y, v_size, dim=0)[rank]

    #print('backward')
    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
        # print('normal loss')
    # if datay_rank.size(0) > 0:
        #print(outputs[rank_train_mask].size())
        #print(datay_rank[rank_train_mask].size()) 
        #print(datay_rank[rank_train_mask])
        loss = 0        
        '''
        print(str(rank_train_mask.size())+ ' ' +str(rank))
        print(str(datay_rank.size())+ ' ' +str(rank))
        print(str(outputs.size())+ ' ' +str(rank))
        print(dim_count)
        '''
        if ht:
            loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask].long())
        else:
            row_id = rank // (size//replication)
            loss = dist_nll(outputs[rank_train_mask], datay_rank[rank_train_mask].long(),row_groups[row_id])
        # loss = F.nll_loss(outputs, torch.max(datay_rank, 1)[1])
        
        loss.backward()
    else:
        print('fake loss')
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        # fake_loss = (outputs * torch.zeros(outputs.size())).sum()
        fake_loss.backward()

    optimizer.step()

    return outputs

def test(outputs, data, vertex_count, rank):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask.bool()].max(1)[1]
        acc = pred.eq(data.y[mask.bool()]).sum().item() / mask.bool().sum().item()
        accs.append(acc)

    if len(accs) == 1:
        accs.append(0)
        accs.append(0)

    return accs
    # logits, accs = outputs, []
    # datay_rank = torch.split(data.y, vertex_count)[rank]
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     mask_rank = torch.split(mask, vertex_count)[rank]
    #     count = mask_rank.nonzero().size(0)
    #     if count > 0:
    #         pred = logits[mask_rank].max(1)[1]
    #         acc = pred.eq(datay_rank[mask_rank]).sum().item() / mask_rank.sum().item()
    #         # pred = logits[mask].max(1)[1]
    #         # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     else:
    #         acc = -1
    #     accs.append(acc)
    # return accs


# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, rank=None, save=False):
    if not normalization:
        return adj_part

    # Scale each edge (u, v) by 1 / (sqrt(u) * sqrt(v))
    # indices = adj_part._indices()
    # values = adj_part._values()

    # deg_map = dict()
    # for i in range(adj_part._nnz()):
    #     u = indices[0][i] + row_vtx
    #     v = indices[1][i] + col_vtx

    #     if u.item() in deg_map:
    #         degu = deg_map[u.item()]
    #     else:
    #         degu = (adj_matrix[0] == u).sum().item()
    #         deg_map[u.item()] = degu

    #     if v.item() in deg_map:
    #         degv = deg_map[v.item()]
    #     else:
    #         degv = (adj_matrix[0] == v).sum().item()
    #         deg_map[v.item()] = degv

    #     values[i] = values[i] / (math.sqrt(degu) * math.sqrt(degv))
    
    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].double(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)
    

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    if save:
        pre = '/scratch/general/nfs1/u1320844/dataset/GNNs'
        torch.save(adj_part, f"{pre}/{graphname}_adj_part_rk{rank}.pt")

    return adj_part

def symmetric(adj_matrix):
    # print(adj_matrix)
    # not sure whether the following is needed
    adj_matrix = adj_matrix.to(torch.device("cpu"))

    adj_matrix, _ = remove_self_loops(adj_matrix)

    # Make adj_matrix symmetrical
    idx = torch.LongTensor([1,0])
    adj_matrix_transpose = adj_matrix.index_select(0,idx)
    # print(adj_matrix_transpose)

    adj_matrix = torch.cat([adj_matrix,adj_matrix_transpose],1)
   
    adj_matrix, _ = add_remaining_self_loops(adj_matrix) 
   
    adj_matrix.to(device)
    return adj_matrix
    
def get_proc_groups(rank, size):
    global replication

    col_count = size // replication

    row_procs = []
    for i in range(0, size, col_count):
        row_procs.append(list(range(i, i + col_count)))

    col_procs = []
    for i in range(col_count):
        col_procs.append(list(range(i, size, col_count)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

def oned_partition(rank, size, inputs, adj_matrix, data, features, classes, device):
    #global row_count
    #global col_count
    '''
    This function will divide sparse matrix into replication
    row panels. Each row panel then will be divided into 
    replication parts so we can do an easier SpMM without merging
    dense matrices. So we are going to reuse CAGNET 1D code here
    only replacing size with replication and changing the initial division
    dimension from vertival to horizontal.
    '''
    global dim_count
    global ht
    global replication
    if rank==0:    print('xxx')
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / replication)
    c_per_proc = math.ceil(float(node_count) / (replication))

    am_partitions = None
    am_pbyp = None

    col_count = size // replication
    row_id = rank // col_count
    col_id = rank %  col_count

    if rank==0:    print('aaa')
    inputs = inputs.to(torch.device("cpu"))
    adj_matrix = adj_matrix.to(torch.device("cpu"))
    if rank==0:    print('bbb')

    # Compute the adj_matrix and inputs partitions for this process
    # TODO: Maybe I do want grad here. Unsure.
    with torch.no_grad():        
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 0)

        if rank==0:    print('ccc')
        r_node_count = vtx_indices[row_id + 1] - vtx_indices[row_id]
        
        am_pbyp, col_indices = split_coo(am_partitions[row_id], node_count, c_per_proc, 1)
        if rank==0:    print('ddd')
        
        for i in range(replication):
            c_node_count = col_indices[i + 1] - col_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                    size=(r_node_count,c_node_count),
                                                    requires_grad=False)

            if rank==0:    print('eee')
            am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[row_id], 
                                                col_indices[i])
            
            if rank==0:    print('fff')
                
        proc_node_count = vtx_indices[row_id + 1] - vtx_indices[row_id]
        am_partitions[row_id] = torch.sparse_coo_tensor(am_partitions[row_id], 
                                                torch.ones(am_partitions[row_id].size(1)), 
                                                size=(proc_node_count, node_count), 
                                                requires_grad=False)
        am_partitions[row_id] = scale_elements(adj_matrix, am_partitions[row_id], node_count,  vtx_indices[row_id], 0, rank=rank,save=True)
        if rank==0:    print('ggg')
        
        # First split inputs into multiple row panels
        inputs_row = torch.split(inputs, math.ceil(float(inputs.size(0)) / replication), dim=0)
        inputs_ = inputs_row[row_id]
        if rank==0:    print('hhh')
        
        if mmorder[0] == 's':
            input_partitions = torch.split(inputs_, math.ceil(float(inputs_.size(1)) / col_count), dim=1)
            ht = False 
        else:
            input_partitions = torch.split(inputs_, math.ceil(float(inputs_.size(0)) / col_count), dim=0)
            ht = True
        adj_matrix_loc = am_partitions[row_id]
        inputs_loc = input_partitions[col_id]

    
    dim_count = []
    for c in range(replication):        
        row_size = min((c+1)*n_per_proc, inputs.size(0)) - c*n_per_proc
        row_tile = math.ceil(float(row_size) / col_count)
        row_cnt = [row_tile]*(col_count-1)
        row_cnt += [row_size - sum(row_cnt)]
        
        col_tile = math.ceil(float(features) / col_count)
        col_cnt = [col_tile] * (col_count-1)
        col_cnt += [features - sum(col_cnt)]
        
        dim_count += [[row_cnt,col_cnt]]
        
            
        
    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    for a in am_pbyp:
        print(f"rank: {rank} adj_matrix_tile_loc.size: {a.size()}", flush=True)
    print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    print(f"rank: {rank} inputs_tile.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, am_partitions[row_id], am_pbyp, ht

def simulate_comm_comp(features,hidden,classes,layers,order):
    comm = 0
    comp = 0
    #Simple code for 2 layers. Can be genralized later
    #Find inter transpose costs    
    for i in range(1,2*layers):
        if order[i] == order[i-1]:
            if i == layers:
                comm += classes
            else:
                comm += hidden
    
    # Find for first and last layer intra transpose
    if order[0] == 'd':
        comm += hidden
        comp += hidden
    else:
        comm += features
        comp += features
    
    if order[1] == 's':
        comm += hidden
        comp += hidden
    else:
        comm += classes
        comp += classes
    
    if order[3] == 's':
        comm += hidden
        comp += hidden
    else:
        comm += features
        comp += features
    
    if order[2] == 'd':
        comm += hidden
        comp += hidden
    else:
        comm += classes
        comp += classes
    
    # Find cases for extra comp and comm for W gradient
    if order[0] == 'd' and order[3] == 'd':
        comm += min(features,hidden)
        comp += min(features,hidden)
    
    if order[1] == 'd' and order[2] == 'd':
        comm += min(classes,hidden)
        comp += min(classes,hidden)
        
    return [comm,comp]

def find_candidates(features,hidden,classes,layers):
    #naive P(4^layer) algorithm. Maybe improved later
    candidates = []
    all_cases = []
    #populate all_cases
    for i in range(4 ** layers):
        c_str = "{0:b}".format(i)
        while len(c_str) < 2*layers:
            c_str = '0' + c_str
        
        c_str = c_str.replace('0','d').replace('1','s')
        if c_str[layers-1] == 's': # Eliminate cases needing distributed loss
            all_cases += [c_str]
    
    comm_comp = []    
    for c in all_cases:
        comm_comp += [simulate_comm_comp(features,hidden,classes,layers,c)+[c]]
    
    comm_comp.sort()
    comm = 0
    comp = comm_comp[0][1]    
    candidates += [comm_comp[0][2]]
    for c in comm_comp:
        if c[1] < comp or (c[0] == comm and c[1] == comp):
            comp = c[1]
            comm = c[0]
            candidates += [c[2]]           
    return candidates

def run(rank, size, inputs, adj_matrix, data, features, classes, device):
    global epochs
    global mid_layer
    global run
    global timing
    global ht

    best_val_acc = test_acc = 0
    outputs = None
    
    if rank==0:    print('777')

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size)

    if rank >= size:
        return

    # adj_matrix_loc = torch.rand(node_count, n_per_proc)
    # inputs_loc = torch.rand(n_per_proc, inputs.size(1))

    # adj_matrix = symmetric(adj_matrix)
   
    if rank==0:    print('888')
    inputs_loc, adj_matrix_loc, am_pbyp, horizontal_tiled = oned_partition(rank, size, inputs, adj_matrix, data, 
                                                                features, classes, device)

    
    if rank==0:    print('999')
    inputs_loc = inputs_loc.to(device)
    #inputs_loc = inputs_loc.half().to(device)
    #adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(am_pbyp)):
        #am_pbyp[i] = am_pbyp[i].coalesce().to(device)
        am_pbyp[i] = am_pbyp[i].coalesce().half().to(device)
    if rank==0:    print('am_pbyp to device')

    for i in range(run_count):
        run = i
        torch.manual_seed(0)
        weight1_nonleaf = torch.rand(features, mid_layer, requires_grad=True)
        weight1_nonleaf = weight1_nonleaf.to(device)
        weight1_nonleaf.retain_grad()

        weight2_nonleaf = torch.rand(mid_layer, classes, requires_grad=True)
        weight2_nonleaf = weight2_nonleaf.to(device)
        weight2_nonleaf.retain_grad()

        weight1 = Parameter(weight1_nonleaf)
        weight2 = Parameter(weight2_nonleaf)

        optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
        dist.barrier(group)

        tstart = 0.0
        tstop = 0.0

        total_time[i] = dict()
        comm_time[i] = dict()
        comp_time[i] = dict()
        scomp_time[i] = dict()
        dcomp_time[i] = dict()
        bcast_comm_time[i] = dict()
        barrier_time[i] = dict()
        barrier_subset_time[i] = dict()
        op1_comm_time[i] = dict()
        op2_comm_time[i] = dict()

        total_time[i][rank] = 0.0
        comm_time[i][rank] = 0.0
        comp_time[i][rank] = 0.0
        scomp_time[i][rank] = 0.0
        dcomp_time[i][rank] = 0.0
        bcast_comm_time[i][rank] = 0.0
        barrier_time[i][rank] = 0.0
        barrier_subset_time[i][rank] = 0.0
        op1_comm_time[i][rank] = 0.0
        op2_comm_time[i][rank] = 0.0

        timing_on = timing == True
        timing = False
        
        candidate_order = [mmorder]
        #candidate_order = find_candidates(features,mid_layer,classes,2)
        print(candidate_order)
        #print(features)
        #print(classes)
        #print(mid_layer)
        order_time = []
        
        ep_start = time.time()
        mmorder_c = candidate_order[0]
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, mmorder_c, data, 
                                rank, size, group, row_groups, col_groups, ht)
        ep_end = time.time()
        ep_time = ep_end - ep_start
        order_time += [ep_time]
        best_mmorder = -1
        
        if timing_on:
            timing = True

        dist.barrier(group)
        tstart = time.time()

        # for epoch in range(1, 201):
        print(f"\nStarting training... rank {rank} run {i}", flush=True)
        elapsed_time = 0
        for epoch in range(1, epochs):            
            if len(order_time) < len(candidate_order):
                mmorder_c = candidate_order[len(order_time)]
            elif best_mmorder == -1:
                best_mmorder = 0
                for i in range(1,len(candidate_order)):
                    if order_time[i] < order_time[best_mmorder]:
                        best_mmorder = i                
                print('Best MM order is '+str(candidate_order[best_mmorder])+' at rank '+str(rank))
            else:
                mmorder_c = candidate_order[best_mmorder]
                        
            ep_start = time.time()            
            outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, mmorder_c, data, 
                                    rank, size, group, row_groups, col_groups, ht)
            ep_end = time.time()
            
            ep_time = ep_end - ep_start
            elapsed_time+=ep_time
            if len(order_time) < len(candidate_order):
                order_time += [ep_time]
                
            if rank == 0: print("\nEpoch: {:03d}".format(epoch), flush=True)
            if accuracy:
                # This is incorrect and needs to be changed accordingly
                # All-gather outputs to test accuracy
                output_parts = []
                n_per_proc = math.ceil(float(inputs.size(0)) / size)
                # print(f"rows: {am_pbyp[-1].size(0)} cols: {classes}", flush=True)
                for i in range(size):
                    output_parts.append(torch.cuda.FloatTensor(n_per_proc, classes, device=device).fill_(0))
                if outputs.size(0) != n_per_proc:
                    pad_row = n_per_proc - outputs.size(0) 
                    outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)
                dist.all_gather(output_parts, outputs)
                output_parts[rank] = outputs
                padding = inputs.size(0) - n_per_proc * (size - 1)
                output_parts[size - 1] = output_parts[size - 1][:padding,:]
                outputs = torch.cat(output_parts, dim=0)
                train_acc, val_acc, tmp_test_acc = test(outputs, data, am_pbyp[0].size(1), rank)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, train_acc, best_val_acc, test_acc)
                if rank == 0:
                    print(log)
                    with open(args.acc_csv, 'a') as f:
                        ws = os.environ['WORLD_SIZE']
                        log = f'{graphname},RDM,{ws},{epoch:03d},{elapsed_time:.4f},{test_acc:.4f}\n'
                        f.write(log) 


        # dist.barrier(group)
        tstop = time.time()
        total_time[i][rank] = tstop - tstart + ep_time

    # Get median runtime according to rank0 and print that run's breakdown
    dist.barrier(group)
    if rank == 0:
        total_times_r0 = [] 
        for i in range(run_count):
            total_times_r0.append(total_time[i][0])

        print(f"total_times_r0: {total_times_r0}")
        median_run_time = statistics.median(total_times_r0)
        median_idx = total_times_r0.index(median_run_time)
        median_idx = torch.cuda.LongTensor([median_idx])
    else:
        median_idx = torch.cuda.LongTensor([0])
        
    dist.broadcast(median_idx, src=0, group=group)        
    median_idx = median_idx.item()
    if rank == 0:
        print(f"rank: {rank} median_run: {median_idx}")
        print(f"rank: {rank} total_time: {total_time[median_idx][rank]}")
        print(f"rank: {rank} comm_time: {comm_time[median_idx][rank]}")
        print(f"rank: {rank} comp_time: {comp_time[median_idx][rank]}")
        print(f"rank: {rank} scomp_time: {scomp_time[median_idx][rank]}")
        print(f"rank: {rank} dcomp_time: {dcomp_time[median_idx][rank]}")
        print(f"rank: {rank} bcast_comm_time: {bcast_comm_time[median_idx][rank]}")
        print(f"rank: {rank} barrier_time: {barrier_time[median_idx][rank]}")
        print(f"rank: {rank} barrier_subset_time: {barrier_subset_time[median_idx][rank]}")
        print(f"rank: {rank} op1_comm_time: {op1_comm_time[median_idx][rank]}")
        print(f"rank: {rank} op2_comm_time: {op2_comm_time[median_idx][rank]}")
        print(f"rank: {rank} {outputs}")
        # total,comm,comp,scomp,dcomp,bcast,reduce,opcomm,barrier
        total = total_time[median_idx][rank] / 10
        comm = comm_time[median_idx][rank]/10
        comp = comp_time[median_idx][rank]/10
        scomp = scomp_time[median_idx][rank]/10
        dcomp = dcomp_time[median_idx][rank]/10
        bcast = bcast_comm_time[median_idx][rank]/10
        red = -1#reduce_comm_time[median_idx][rank]/10
        opcomm = -1# op_comm_time[median_idx][rank]/10
        bar = barrier_time[median_idx][rank]/10
        ws = os.environ['WORLD_SIZE']
        stats = f'RDM,{graphname},{mmorder},{mid_layer},{ws},{total},{comm},{comp},{scomp},{dcomp},{bcast},{red},{opcomm},{bar}\n'
        #with open('stats_3.csv','a') as f:
        #with open('stats_mm_np_modelvalidation.csv'.format(mid_layer),'a') as f:
        with open('../dgl/sep23_data/stats_2nodes.csv','a') as f:
            f.write(stats)
        print('DOne')
        exit()



    
    
    if accuracy:
        # This is incorrect and needs to be changed accordingly
        # All-gather outputs to test accuracy
        output_parts = []
        n_per_proc = math.ceil(float(inputs.size(0)) / size)
        # print(f"rows: {am_pbyp[-1].size(0)} cols: {classes}", flush=True)
        for i in range(size):
            output_parts.append(torch.cuda.FloatTensor(n_per_proc, classes, device=device).fill_(0))

        if outputs.size(0) != n_per_proc:
            pad_row = n_per_proc - outputs.size(0) 
            outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)

        dist.all_gather(output_parts, outputs)
        output_parts[rank] = outputs
        
        padding = inputs.size(0) - n_per_proc * (size - 1)
        output_parts[size - 1] = output_parts[size - 1][:padding,:]

        outputs = torch.cat(output_parts, dim=0)

        train_acc, val_acc, tmp_test_acc = test(outputs, data, am_pbyp[0].size(1), rank)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

        print(log.format(900, train_acc, best_val_acc, test_acc))

    return outputs

def rank_to_devid(rank, acc_per_rank):
    return rank % acc_per_rank

def init_process(rank, size, inputs, adj_matrix, data, features, classes, device, outputs, fn):
    run_outputs = fn(rank, size, inputs, adj_matrix, data, features, classes, device)
    if outputs is not None:
        outputs[rank] = run_outputs.detach()

def main():
    global device
    global graphname

    print(socket.gethostname())
    seed = 0

    if not download:
        #mp.set_start_method('spawn', force=True)
        mp.set_start_method('fork', force=True)
        outputs = None
        '''
        if "OMPI_COMM_WORLD_RANK" in os.environ.keys():
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

        # Initialize distributed environment with SLURM
        if "SLURM_PROCID" in os.environ.keys():
            os.environ["RANK"] = os.environ["SLURM_PROCID"]

        if "SLURM_NTASKS" in os.environ.keys():
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

        if "MASTER_ADDR" not in os.environ.keys():
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        os.environ["MASTER_PORT"] = "1234"
        '''
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        size = dist.get_world_size()
        if 2 ** round(math.log2(size / replication)) != size//replication:
            print('ERROR: size/replication must be a power of 2')
            exit()
            
        print("Processes: " + str(size))

        # device = torch.device('cpu')
        devid = rank_to_devid(rank, acc_per_rank)
        device = torch.device('cuda:{}'.format(devid))
        print(f"device: {device}")
        torch.cuda.set_device(device)
        if rank==0:    print("000")
        curr_devid = torch.cuda.current_device()
        # print(f"curr_devid: {curr_devid}", flush=True)
        devcount = torch.cuda.device_count()

    if graphname == "Cora":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
        dataset = Planetoid(path, graphname, transform=T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == "Reddit":
        path = '/scratch/general/nfs1/u1320844/dataset/reddit'
        dataset = Reddit(path, T.NormalizeFeatures())
        data = dataset[0]
        data = data.to(device)
        data.x.requires_grad = True
        inputs = data.x.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif graphname == 'Amazon':
        # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', graphname)
        # edge_index = torch.load(path + "/processed/amazon_graph.pt")
        # edge_index = torch.load("/gpfs/alpine/bif115/scratch/alokt/Amazon/processed/amazon_graph_jsongz.pt")
        # edge_index = edge_index.t_()
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../data/Amazon/processed/data.pt")
        print(f"Done loading coo", flush=True)
        # n = 9430088
        n = 14249639
        num_features = 300
        num_classes = 24
        # mid_layer = 24
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        # edge_index = edge_index.to(device)
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        print(f"edge_index: {edge_index}", flush=True)
        data = data.to(device)
        # inputs = inputs.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname == 'subgraph3':
        # path = "/gpfs/alpine/bif115/scratch/alokt/HipMCL/"
        # print(f"Loading coo...", flush=True)
        # edge_index = torch.load(path + "/processed/subgraph3_graph.pt")
        # print(f"Done loading coo", flush=True)
        print(f"Loading coo...", flush=True)
        edge_index = torch.load("../data/subgraph3/processed/data.pt")
        print(f"Done loading coo", flush=True)
        n = 8745542
        num_features = 128
        # mid_layer = 512
        # mid_layer = 64
        num_classes = 256
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes - 1).long()
        data.train_mask = torch.ones(n).long()
        print(f"edge_index.size: {edge_index.size()}", flush=True)
        data = data.to(device)
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname in ['meta', 'arctic25', 'oral', 'airways']:
        pref = '/scratch/general/nfs1/u1320844/dataset/asplos/{}_subgs/'.format(graphname)
        adj_full = torch.load(pref+'adj_full.pt')
        x_full = torch.load(pref+'x_full.pt')
        y_full = torch.load(pref+'y_full.pt')
        print(f'{graphname}, {len(adj_full[0])}, {len(x_full)}, {len(adj_full[0])/len(x_full)}')
#        exit()

        train_mask_full = torch.load(pref+'train_mask_full.pt')
        val_mask_full = torch.load(pref+'val_mask_full.pt')
        test_mask_full = torch.load(pref+'test_mask_full.pt')
        data = Data()
        data.edge_index = adj_full
        data.x = x_full
        data.y = y_full
        data.x.requires_grad = True
        data.train_mask = train_mask_full
        data.val_mask = val_mask_full
        data.test_mask = test_mask_full
        inputs = data.x.to(device)
        data.y = data.y.to(device)
        edge_index = data.edge_index
        num_features = x_full.shape[-1]
        tmp = {}
        tmp['meta'] = 25
        tmp['airways'] = 25
        tmp['oral'] = 32 
        tmp['arctic25'] = 33
        num_classes = tmp[graphname]
    elif 'ogb' in graphname:
        path = '/scratch/general/nfs1/u1320844/dataset'
        if rank==0:    print('Building dataset...')
        dataset = PygNodePropPredDataset(graphname, path)
        if rank==0:    print('111')
        #evaluator = Evaluator(name=graphname)
        if 'mag' in graphname:
            rel_data = dataset[0]
            # only train with paper <-> paper relations.
            data = Data(
                x=rel_data.x_dict['paper'],
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            data = T.NormalizeFeatures()(data)
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']['paper']
            val_idx = split_idx['valid']['paper']
            test_idx = split_idx['test']['paper']
        else:
            split_idx = dataset.get_idx_split()
            data = dataset[0]
            #data = data.to(device)
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']
            if rank==0:    print('222')

        data.x.requires_grad = True
        inputs = data.x#.to(device).half()
        if rank==0:    print('333')
        #inputs.requires_grad = True
        data.y = data.y.squeeze().to(device)
        edge_index = data.edge_index
        if rank==0:    print('444')
        if 'arxiv' in graphname:
            edge_index = symmetric(edge_index)
        if rank==0:    print('555')
        num_features = dataset.num_features if not 'mag' in graphname else 128
        num_classes = dataset.num_classes
        num_nodes = len(data.x)
        train_mask = torch.zeros(num_nodes)
        train_mask[train_idx] = 1
        val_mask = torch.zeros(num_nodes)
        val_mask[val_idx] = 1
        test_mask = torch.zeros(num_nodes)
        test_mask[test_idx] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        if rank==0:    print('666')

    elif graphname == 'com-orkut':
        edge_index = torch.load('/scratch/general/nfs1/u1320844/dataset/com_orkut/com-orkut.pt')
        print(f"Done loading coo", flush=True)
        n = 3072441
        n = 3072627
        num_features = 128
        num_classes = 100
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes-1).long()
        data.train_mask = torch.ones(n).long()
        inputs.requires_grad = True
        data.y = data.y.to(device)
    elif graphname == 'web-google':
        edge_index = torch.load('/scratch/general/nfs1/u1320844/dataset/web_google/web-Google.pt')
        print(f"Done loading coo", flush=True)
        n = 916428
        num_features = 256
        num_classes = 100
        inputs = torch.rand(n, num_features)
        data = Data()
        data.y = torch.rand(n).uniform_(0, num_classes-1).long()
        data.train_mask = torch.ones(n).long()
        inputs.requires_grad = True
        data.y = data.y.to(device)

    if download:
        exit()

    if normalization:
        adj_matrix, _ = add_remaining_self_loops(edge_index, num_nodes=inputs.size(0))
    else:
        adj_matrix = edge_index


    if rank==0:    print('initing process')
    init_process(rank, size, inputs, adj_matrix, data, num_features, num_classes, device, outputs, 
                    run)

    if outputs is not None:
        global ht
        res = outputs[0]
        if not ht:
            res = transpose_input(res,rank,size,1)
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--accperrank", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--graphname", type=str)
    parser.add_argument("--timing", type=str)
    parser.add_argument("--midlayer", type=int)
    parser.add_argument("--runcount", type=int)
    parser.add_argument("--normalization", type=str)
    parser.add_argument("--activations", type=str)
    parser.add_argument("--accuracy", type=str)
    parser.add_argument("--download", type=bool)
    parser.add_argument("--mmorder", type=str)
    parser.add_argument("--replication", type=int)
    parser.add_argument("--acc_csv", type=str)


    args = parser.parse_args()
    print(args)

    acc_per_rank = args.accperrank

    epochs = args.epochs
    graphname = args.graphname
    timing = args.timing == "True"
    mid_layer = args.midlayer
    run_count = args.runcount
    normalization = args.normalization == "True"
    activations = args.activations == "True"
    accuracy = args.accuracy == "True"
    download = args.download
    mmorder = args.mmorder
    replication = args.replication
   
    if mmorder == None or mmorder == "":
        mmorder = "dsds"

    if replication == 0 or replication == None:
        replication = 1
        

    if not download:
        if (epochs is None) or (graphname is None) or (timing is None) or (mid_layer is None) or (run_count is None):
            print(f"Error: missing argument {epochs} {graphname} {timing} {mid_layer} {run_count}")
            exit()

    print(f"Arguments: epochs: {epochs} graph: {graphname} timing: {timing} mid: {mid_layer} norm: {normalization} act: {activations} acc: {accuracy} runs: {run_count} MM order: {mmorder} replication {replication}")
    
    print(main())
