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
    version='0.8.1',
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
        "OpenMM>=8.4,<8.5",
        "MDAnalysis>=2.10,<2.11",
        "mdtraj>=1.11,<1.12",
        'numpy',
        'pandas',
        'biopython',
        'Jinja2',
        'tqdm',
        'matplotlib',
        'PyYAML',
        'statsmodels',
        'localcider',
        'pytest',
        'numba',
        'scipy'
    ],
    python_requires=">=3.13,<3.14",

    # include_package_data=True,
    package_data={'' : ['data/*.csv', 'data/*.yaml', 'data/templates/*']},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.13',
    ],
)
