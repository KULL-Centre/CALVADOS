The lines below run simulations of an IDR with an MDP:

```bash
python prepare.py --name_1 <name protein 1> --name_2 <name protein 2>
python <name protein 1>_<name protein 2>/run.py --path <name protein 1>_<name protein 2>
```

where `<name protein 1>` (Tau35) is an IDR and `<name protein 2>` (TIA1) is an MDP with sequence and structure provided in the `input` folder.

After running the simulation, the script analyses the trajectory and saves the following files in the `data` folder:

```bash
com_traj.dcd # trajectory of the centers of mass
com_top.pdb # pdb file of the centers of mass
Tau35_TIA1_Tau35_TIA1_cmap.npy # Tau35-TIA1 contact map
Tau35_TIA1_Tau35_rg.npy # per-frame Rg's of Tau35
Tau35_TIA1_TIA1_rg.npy # per-frame Rg's of TIA1
```

Trajectory of the centers of mass can be used to calculate the radial distribution function and the second virial coefficient

```python
import mdtraj as md
import numpy as np

t = md.load('data/com_traj.dcd',top='data/com_top.pdb')
r,rdf = md.compute_rdf(t,pairs=[[0,1]],r_range=(.5,15),bin_width=.1)
b22 = -2*np.pi*np.trapz((rdf-1)*r*r,r)
```
