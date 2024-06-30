from analyse import *
import os
import subprocess
from jinja2 import Template

sequences = init_proteins()
sequences.to_csv('sequences.csv')

submission = Template("""#!/bin/sh

conda activate calvados

python ./simulate.py --name {{name}} --temp {{temp}}""")

for name in ['A1']:
    if not os.path.isdir(name):
        os.mkdir(name)
    for temp in [293]:
        if not os.path.isdir(name+'/{:d}'.format(temp)):
            os.mkdir(name+'/{:d}'.format(temp))
        with open('{:s}_{:d}.sh'.format(name,temp), 'w') as submit:
            submit.write(submission.render(name=name,temp='{:d}'.format(temp)))
        subprocess.run(['sh','{:s}_{:d}.sh'.format(name,temp)])
