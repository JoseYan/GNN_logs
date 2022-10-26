graph=$1
out=$2

if [ $graph == "arctic25" ]
then 
    nnodes=500000
else
    nnodes=1000000 
fi
for loader in node edge rw
do
    for bs in 1000 2000 4000 8000
    do
        for lr in 0.01 0.001 0.0001 0.1
        do 
            for dr in 0 0.1 0.2 0.3
            do
                nsubg=$((nnodes/bs))
                cmd="python meta_gnn_overlap_sample.py -i x -n $graph -o $out -l $loader -b $bs -s $nsubg --load --lr $lr --dropout $dr"
                echo $cmd >> run_$graph.sh
            done
        done
    done
done
