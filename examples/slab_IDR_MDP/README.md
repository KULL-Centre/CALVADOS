The lines below runs a slab simulation of an IDR (100 copies) in the presence of an MDP (100 copies):

```bash
python prepare.py --name_1 <name protein 1> --name_2 <name protein 2>
python <name protein 1>_<name protein 2>/run.py --path <name protein 1>_<name protein 2>
```

where `<name protein 1>` (Tau35) is an IDR and `<name protein 2>` (TIA1) is an MDP with sequence and structure provided in the `input` folder.

After running the simulation, the script analyses the trajectory and saves the following files in the `data` folder:

```bash
Tau35_TIA1_com_traj.dcd # trajectory of the centers of mass
Tau35_TIA1_com_top.pdb # pdb file of the centers of mass
Tau35_TIA1_Tau35_TIA1_cmap.npy # Tau35-TIA1 contact map
Tau35_TIA1_Tau35_rg.npy # per-frame Rg's of Tau35
Tau35_TIA1_TIA1_rg.npy # per-frame Rg's of TIA1
Tau35_TIA1_Tau35_rg_dilute.npy # Rg's of A1 LCD in the dilute phase
Tau35_TIA1_Tau35_rg_dense.npy # Rg's of A1 LCD in the dense phase
Tau35_TIA1_ps_results.csv # Average concentrations in the protein-rich and protein-dilute phases
Tau35_TIA1_Tau35_dilute_array.npy # time series of the protein concentration in the dilute phase
Tau35_TIA1_Tau35_dense_array.npy # time series of the protein concentration in the dense phase
Tau35_TIA1_TIA1_dilute_array.npy # time series of the protein concentration in the dilute phase
Tau35_TIA1_TIA1_dense_array.npy # time series of the protein concentration in the dense phase
Tau35_TIA1_profiles.npy # contains three columns: z-coordinate, average Tau35-concentration profile, average TIA1-concentration profile
Tau35_TIA1_Tau35_profile.npy # per-frame Tau35-concentration profiles
Tau35_TIA1_TIA1_profile.npy # per-frame TIA1-concentration profiles
Tau35_TIA1_Tau35_Tau35_homotypic_cmap.npy # Tau35-Tau35 contact map
```
