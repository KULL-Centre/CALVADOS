The lines below run direct coexistence simulations of 100 copies of A1 LCD in PEG x % w/v

```bash
python prepare.py --mw <PEG MW> --wv <PEG g / 100 mL> --gpu_id <GPU ID>
python PEG<PEG MW>_<PEG g / 100 mL>/run.py --path PEG<PEG MW>_<PEG g / 100 mL>
```

For example, for 5 % w/v PEG400 

```bash
python prepare.py --mw 400 --wv 5 --gpu_id 0
python PEG400_5.00/run.py --path PEG400_5.00
```

After running the simulation, the script analyses the trajectory and generates the following files in the `data` folder:

```bash
PEG400_5_A1_rg_dilute.npy # Rg's of A1 LCD in the dilute phase
PEG400_5_A1_rg_dense.npy # Rg's of A1 LCD in the dense phase
PEG400_5_ps_results.csv # Concentrations of A1 LCD and PEG in the protein-rich and protein-dilute phases
PEG400_5_A1_dilute_array.npy # time series of the protein concentration in the dilute phase
PEG400_5_A1_dense_array.npy # time series of the protein concentration in the dense phase
PEG400_5_PEG400_dilute_array.npy # time series of the PEG concentration in the dilute phase
PEG400_5_PEG400_dense_array.npy # time series of the PEG concentration in the dense phase
PEG400_5_profiles.npy # contains three columns: z-coordinate, average protein-concentration profile, average PEG-concentration profile
PEG400_5_A1_profile.npy # per-frame protein-concentration profiles
PEG400_5_PEG_profile.npy # per-frame PEG-concentration profiles
PEG400_5_A1_A1_homotypic_cmap.npy # A1-A1 contact map
```
