import subprocess
import os
import time
import shutil
from jinja2 import Template

submission = Template("""#!/bin/sh
export CUDA_VISIBLE_DEVICES={{gpu_id}}
python prepare.py --name {{name}} --gpu_id {{gpu_id}} --replica {{replica}} --box_size {{box_size}}
python {{name}}_{{replica}}/run.py --path {{name}}_{{replica}}
""")

for name,box_size in zip(['A1noNLS'],[16.454]):
    for replica, gpu_id in zip([0],[0]):
        folder = f'{name:s}_{replica:d}'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        with open(folder+'/submit.sh', 'w') as submit:
            submit.write(submission.render(name=name,replica=replica,gpu_id=gpu_id,box_size=box_size))
        subprocess.run(f'nohup sh {folder}/submit.sh > {folder}.out 2> {folder}.err &',shell=True)
        print(folder)
        time.sleep(.6)
