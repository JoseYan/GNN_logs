mm=dsds
st="/uufs/chpc.utah.edu/common/home/u1320844/GNNs/dgl/oct6_data/stats_1node.csv"
for graph in com-orkut  web-google
do
    for np in 2 4 8
    do
        torchrun --nproc_per_node=$np --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=$np --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
    done
done
for graph in com-orkut  web-google
do
    for np in 2 4 8
    do
        torchrun --nproc_per_node=$np --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12399 src/gcn_distr_15d.py --accperrank=$np --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --graphname=$graph --stats=$st
    done
done
mm=sdsd
for graph in com-orkut  web-google
do
    for np in 2 4 8
    do
        torchrun --nproc_per_node=$np --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=$np --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=1 --mmorder=$mm --graphname=$graph --stats=$st
    done
done