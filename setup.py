from setuptools import setup
import os
import subprocess

# download BLOCKING code from GitHub repo
if not os.path.isdir('calvados/BLOCKING'):
    os.mkdir('calvados/BLOCKING')
subprocess.run(['curl','-o','calvados/BLOCKING/main.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/main.py'])
subprocess.run(['curl','-o','calvados/BLOCKING/block_tools.py','https://raw.githubusercontent.com/fpesceKU/BLOCKING/v0.1/block_tools.py'])

setup(
    name='calvados',
    version='0.4.0',
    description='Coarse-grained implicit-solvent simulations of biomolecules',
    url='https://github.com/sobuelow/calvados',
    authors = [
        {name='Sören von Bülow', email='soren.bulow@bio.ku.dk'},
        {name='Giulio Tesei', email='giulio.tesei@bio.ku.dk'},
        {name='Fan Cao', email='fan.cao@bio.ku.dk'},
        {name='Kresten Lindorff-Larsen', email='lindorff@bio.ku.dk'}
    ]
    license='GNU GPL3',
    packages=['calvados'],
    install_requires=[
        'numpy',
        'pandas',
        'OpenMM',
        'mdtraj',
        'MDAnalysis',
        'scipy',
        'biopython',
        'Jinja2',
        'progressbar2',
        'matplotlib',
        'numba',
        'PyYAML',
        'statsmodels',
        'localcider'
    ],

    # include_package_data=True,
    package_data={'' : ['data/*.csv', 'data/*.yaml', 'data/templates/*']},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3>=3.7,<3.11',
    ],
)
