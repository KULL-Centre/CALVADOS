The lines below run simulations of a single transmembrane protein in a lipid bilayer where the protein is modeled using CALVADOS 3:

```bash
python prepare.py --name <protein name> --replica <replica index>
python <protein name>_<replica index>/run.py --path <protein name>_<replica index>
```

