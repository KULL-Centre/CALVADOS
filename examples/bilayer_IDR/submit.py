import subprocess
import os
import pandas as pd
import numpy as np
import mdtraj as md
import time
import shutil
from jinja2 import Template

submission = Template("""#!/bin/sh
#SBATCH --job-name={{path}}
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH -t 48:00:00
#SBATCH -o {{path}}/out
#SBATCH -e {{path}}/err

source /home/gitesei/.bashrc
module load gcc/11.2.0 openmpi/4.0.3 cuda/11.2.0
conda activate calvados

echo $SLURM_CPUS_PER_TASK

echo $SLURM_CPUS_ON_NODE

python prepare.py --name {{name}} --N_prot {{N_prot}}
python {{path}}/run.py --path {{path}}""")

for name, N_prot in zip(['FUS'],[40]):
    path = f'{name:s}_{N_prot:d}'
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(path+'/submit.sh', 'w') as submit:
        submit.write(submission.render(name=name,N_prot=N_prot,path=path))
    subprocess.run(['sbatch',path+'/submit.sh'])
    print(path)
    time.sleep(.6)
