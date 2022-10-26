rk=$1
python -m torch.distributed.run --nproc_per_node=2 --nnodes=4 --node_rank=$rk --master_addr=10.242.76.113 --master_port=12399 src/gcn_distr_transpose_15d.py --accperrank=2 --epochs=10  --timing=True --midlayer=128 --runcount=1 --activations=True --replication=2 --mmorder=sdsd --graphname=ogbn-papers100M
