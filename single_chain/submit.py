import os
import subprocess
from jinja2 import Template
import pandas as pd
import numpy as np
import time

submission = Template("""#!/bin/sh
#SBATCH --job-name={{name}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=sbinlab_ib2
#SBATCH --mem=1GB
#SBATCH -t 20:00:00
#SBATCH -o {{name}}.err
#SBATCH -e {{name}}.out

source /groups/sbinlab/giulio/.bashrc
conda activate calvados

python ./simulate.py --name {{name}} --cutoff {{cutoff}}""")

def initProteins():
    proteins = pd.DataFrame(columns=['temp','pH','ionic','fasta'],dtype=object)

    # DOI: 10.1073/pnas.1322611111
    fasta_ACTR = """GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD""".replace('\n', '') 

    proteins.loc['ACTR'] = dict(temp=278,pH=7.4,fasta=list(fasta_ACTR),ionic=0.2)
    return proteins

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

r = pd.read_csv('residues.csv').set_index('three')
r.lambdas = r['CALVADOS2'] # select CALVADOS1 or CALVADOS2 stickiness parameters
r.to_csv('residues.csv')
cutoff = 2.0 # set the cutoff for the nonionic interactions

for name in proteins.index:
    if not os.path.isdir(name):
        os.mkdir(name)
    with open('{:s}.sh'.format(name), 'w') as submit:
        submit.write(submission.render(name=name,cutoff='{:.1f}'.format(cutoff)))
    subprocess.run(['sbatch','{:s}.sh'.format(name)])
