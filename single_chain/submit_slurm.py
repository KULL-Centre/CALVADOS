import os
import subprocess
from jinja2 import Template
import pandas as pd

submission = Template("""#!/bin/sh
#SBATCH --job-name={{seq_name}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=sbinlab_ib
#SBATCH --mem=3GB
#SBATCH -t 72:00:00
#SBATCH -o {{path}}/out
#SBATCH -e {{path}}/err

conda activate calvados

python ./simulate.py --seq_name {{seq_name}} --path {{path}}""")

def init_proteins():
    proteins = pd.DataFrame(columns=['temp','pH','ionic','fasta','N'],dtype=object)

    # DOI: 10.1073/pnas.1322611111
    fasta_ACTR = """GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD""".replace('\n', '')

    proteins.loc['ACTR'] = dict(temp=278,pH=7.4,fasta=fasta_ACTR,ionic=0.2,N=len(fasta_ACTR))
    return proteins

sequences = init_proteins()
sequences.to_csv('sequences.csv')

parent_dir = './' # or e.g. 'replica_0'

if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)

for seq_name in sequences.index:
    path = f'{parent_dir:s}/{seq_name:s}'
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(f'{seq_name:s}.sh', 'w') as submit:
        submit.write(submission.render(seq_name=seq_name,path=path))
    subprocess.run(['sbatch',f'{seq_name:s}.sh'])
