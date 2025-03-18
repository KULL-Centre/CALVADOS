The lines below run simulations of a single IDR:

```bash
python prepare.py --name_1 <name protein 1> --name_2 <name protein 2>
python <name protein 1>_<name protein 2>/run.py --path <name protein 1>_<name protein 2>
```

where `<name protein 1>` (aSyn) and `<name protein 2>` (Tau35) are proteins with sequences provided in the fasta file in the `input` folder.
