mm=sdsd
st="/uufs/chpc.utah.edu/common/home/u1320844/GNNs/dgl/oct6_data/stats.csv"
rk=$1
for graph in com-orkut web-google
do
    CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=1 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
    CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=2 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
    CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=4 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=8 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
done
#for graph in ogbn-arxiv Reddit ogbn-products ogbn-mag airways arctic25 oral
#do
#    CUDA_VISIBLE_DEVICES="0"   torchrun --nproc_per_node=1 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_15d.py --accperrank=1 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --graphname=$graph  --stats=$st
#    CUDA_VISIBLE_DEVICES="0,1"   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_15d.py --accperrank=2 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --graphname=$graph --stats=$st
#    CUDA_VISIBLE_DEVICES="0,1,2,3"   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_15d.py --accperrank=4 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --graphname=$graph  --stats=$st
#    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_15d.py --accperrank=8 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --graphname=$graph --stats=$st
#done