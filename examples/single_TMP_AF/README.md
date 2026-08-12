The lines below run simulations of a single transmembrane protein in a lipid bilayer where the protein is modeled using AF-CALVADOS:

```bash
python fetch_AFDB.py <UniProt ID> <first residue> <last residue> --tmd <first TMD residue> <last TMD residue> --orientation 0 1 0
python prepare.py --name <gene name> --replica <replica index>
python <gene name>_<replica index>/run.py --path <gene name>_<replica index>
```

`fetch_AFDB.py` retrieves an AlphaFold2 model for a protein of given `<gene name>` and `<UniProt ID>`. It selects the structure within the given range of residues and crops the corresponding pLDDT scores and PAEs. Finally, `crop_AFDB.py` orients the TMD so that its principal axis is parallel to the given orientation.

For the non-selective voltage-gated ion channel VDAC1:

```bash
python fetch_AFDB.py P21796 1 283 --tmd 25 283 --orientation 0 1 0
python prepare.py --name VDAC1 --replica 0
python VDAC1_0/run.py --path VDAC1_0
```
