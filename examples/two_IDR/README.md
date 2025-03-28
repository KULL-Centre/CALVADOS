The lines below run simulations of two different IDRs:

```bash
python prepare.py --name_1 <name protein 1> --name_2 <name protein 2>
python <name protein 1>_<name protein 2>/run.py --path <name protein 1>_<name protein 2>
```

where `<name protein 1>` (aSyn) and `<name protein 2>` (Tau35) are proteins with sequences provided in the fasta file in the `input` folder.

After running the simulation, the script analyses the trajectory and saves the following files in the `data` folder:

```bash
aSyn_Tau35_com_traj.dcd # trajectory of the center of mass of aSyn and Tau35
aSyn_Tau35_com_top.pdb # pdb file of the center of mass of aSyn and Tau35
aSyn_Tau35_aSyn_Tau35_cmap.npy # aSyn-Tau35 contact map
aSyn_Tau35_aSyn_rg.npy # per-frame Rg's of aSyn
aSyn_Tau35_Tau35_rg.npy # per-frame Rg's of Tau35
```
