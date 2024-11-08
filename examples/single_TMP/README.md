The lines below run simulations of a single transmembrane protein in a lipid bilayer:

```bash
python prepare.py --name <protein name>
python <protein name>/run.py --path <protein name>
```

where `<protein name>` (hGHR) is a protein for which a PDB file and the residues of the folded domains are provided in the `input` folder.
In the example, we simulate a construct of the human growth hormone receptor, placing the transmembrane helix in the midplane of a lipid bilayer.
