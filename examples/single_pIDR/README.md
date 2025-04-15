The lines below run simulations of a single IDR in phosphorylated state:

```bash
python prepare.py --name <protein name> --pH <solution pH>
python <protein name>/run.py --path <protein name>
```

where `<protein name>` (10pAsh1 or Ash1) is a protein with sequence provided in the fasta file in the `input` folder. Phosphorylated Ser and Thr are represented as B and O in the fasta file. The charge of pSer and pThr is determined based on the input pH value as detailed in [DOI: 10.1101/2025.03.19.644261](https://doi.org/10.1101/2025.03.19.644261).
