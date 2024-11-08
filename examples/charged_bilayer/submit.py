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

python prepare.py --name {{name}} --alpha {{alpha}}
python {{path}}/run.py --path {{path}}""")

for name,alpha in zip(['POPC']*3,[0.1,0.2,0.4]):
    if not os.path.isdir(f'{name:s}_{alpha:.2f}'):
        os.mkdir(f'{name:s}_{alpha:.2f}')
    with open(f'{name:s}_{alpha:.2f}/submit.sh', 'w') as submit:
        submit.write(submission.render(name=name,alpha=alpha,path=f'{name:s}_{alpha:.2f}'))
    subprocess.run(['sbatch',f'{name:s}_{alpha:.2f}/submit.sh'])
    print(f'{name:s}_{alpha:.2f}')
    time.sleep(.6)
