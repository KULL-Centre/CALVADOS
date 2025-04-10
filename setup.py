from setuptools import setup, find_packages
import subprocess

# download BLOCKING code from GitHub repo
try:
    subprocess.run(['wget','-O','calvados/BLOCKING/main.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/main.py'])
    subprocess.run(['wget','-O','calvados/BLOCKING/block_tools.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/block_tools.py'])
except FileNotFoundError:
    subprocess.run(['curl','-o','calvados/BLOCKING/main.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/main.py'])
    subprocess.run(['curl','-o','calvados/BLOCKING/block_tools.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/block_tools.py'])

setup(
    name='calvados',
    version='0.7.0',
    description='Coarse-grained implicit-solvent simulations of biomolecules',
    url='https://github.com/KULL-Centre/CALVADOS',
    authors=[
        {'name':'Soren von Bulow', 'email':'soren.bulow@bio.ku.dk'},
        {'name':'Giulio Tesei', 'email':'giulio.tesei@bio.ku.dk'},
        {'name':'Fan Cao', 'email':'fan.cao@bio.ku.dk'},
        {'name':'Kresten Lindorff-Larsen', 'email':'lindorff@bio.ku.dk'}
    ],
    license='GNU GPL3',
    packages=find_packages(),
    install_requires=[
        'OpenMM==8.2.0',
        'numpy==1.24',
        'pandas==2.1.1',
        'MDAnalysis==2.6.1',
        'biopython==1.81',
        'Jinja2==3.1.2',
        'tqdm==4.66',
        'matplotlib==3.8',
        'PyYAML==6.0',
        'statsmodels==0.14',
        'localcider==0.1.21',
        'pytest==8.3.3',
        'numba==0.60',
        'scipy==1.13',
        'mdtraj==1.10',
    ],

    # include_package_data=True,
    package_data={'' : ['data/*.csv', 'data/*.yaml', 'data/templates/*']},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3>=3.7,<3.11',
    ],
)
