import sys
import os

log_file = sys.argv[1]
info = log_file.strip().split('/')[-1].split('_')
dataset = info[0]
loader = info[3]
bs = info[4]
nsubg = info[5]
lr = info[6]
dr = info[7].split('.log')[0]

shell_script = f"run_{dataset}.sh"
lines = []
with open(shell_script, 'r') as f:
    lines = f.readlines()
last_cmd = f"python meta_gnn_overlap_sample.py -i x -n {dataset} -o log_1026/ -l {loader} -b {bs} -s {nsubg} --load --lr {lr} --dropout {dr}\n"
idx = lines.index(last_cmd)
print('Resuming from:', idx+1)
for cmd in lines[idx+1:]:
    os.system(cmd)


