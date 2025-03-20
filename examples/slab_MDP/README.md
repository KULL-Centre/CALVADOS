The lines below run direct coexistence simulations of 100 copies of a single MDP:

```bash
python prepare.py --name <protein name>  --replica <replica>
python <protein name>_<replica>/run.py --path <protein name>_<replica>
```

where `<protein name>` (hnRNPA1S) is a protein with sequence provided in the `input` folder. To estimate the sampling error of phase properties, we recommend running independent replicas of the same system. `<replica>` is an integer that indicates the replica to simulate.
