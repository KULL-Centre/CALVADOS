The lines below run simulations of full-length A1 with custom restraints between folded domains:

```bash
python prepare.py --name <protein name>
python <protein name>/run.py --path <protein name>
```

where `<protein name>` (hnRNPA1S) is a protein for which a PDB file and the residues of the folded domains are provided in the `input` folder.
