The lines below run simulations of a single multi-domain protein:

```bash
python prepare.py --name <protein name>
python <protein name>/run.py --path <protein name>
```

where `<protein name>` (hnRNPA1S or TIA1) is a protein for which a PDB file and the residues of the folded domains are provided in the `input` folder.
