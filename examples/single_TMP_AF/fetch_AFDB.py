#!/usr/bin/env python3

import argparse
import json
import requests
import mdtraj as md
import numpy as np
from mdtraj.utils.rotation import rotation_matrix_from_quaternion

def download_json(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def download_pdb(url, outfile):
    r = requests.get(url)
    r.raise_for_status()
    with open(outfile, "w") as f:
        f.write(r.text)

def align_axis(traj, v, axis, cm):
    u = np.asarray(axis, dtype=float)
    u /= np.linalg.norm(u)

    v = np.asarray(v, dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    a = np.cross(v, u)
    anorm = np.linalg.norm(a, axis=1, keepdims=True)

    parallel = anorm[:, 0] < 1e-8
    a[~parallel] /= anorm[~parallel]

    b = np.arccos(np.clip(np.dot(v, u), -1.0, 1.0))

    quaternion = np.insert(np.sin(-b/2).reshape(-1, 1) * a, 0, np.cos(-b/2), axis=1)
    quaternion[parallel] = np.array([1.0, 0.0, 0.0, 0.0])

    xyz = traj.xyz - cm.reshape(-1, 1, 3)
    xyz = np.matmul(xyz, rotation_matrix_from_quaternion(quaternion))

    return md.Trajectory(
        xyz,
        topology=traj.top,
        unitcell_lengths=traj.unitcell_lengths,
        unitcell_angles=traj.unitcell_angles,
    )

def write_pdb_with_bfactors(pdb_in, pdb_out, t, atom_indices):
    xyz = t.xyz[0] * 10.0
    atom_indices = set(atom_indices)
    old_to_new = {old: new for new, old in enumerate(sorted(atom_indices))}

    i = 0
    with open(pdb_in) as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if i in atom_indices:
                    x, y, z = xyz[old_to_new[i]]
                    fout.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                i += 1
            elif line.startswith(("TER", "END")):
                fout.write(line)
        fout.write("\n")

def align_pdb(gene, start, end, indexes, orientation=[0, 0, 1]):
    pdb_in = f"input/{gene}_FL.pdb"
    t = md.load_pdb(pdb_in)

    # TMD indices refer to the full-length PDB / UniProt numbering
    d_tmd = t.top.select(f"residue >= {indexes[0] - 1} and residue <= {indexes[1] - 1}")
    td = t.atom_slice(d_tmd)

    cm = md.compute_center_of_mass(td)
    I = md.compute_inertia_tensor(td)
    v = np.linalg.eigh(I)[1][:, :, 0].reshape(-1, 3)

    t = align_axis(t, v, orientation, cm)

    # crop after alignment; start/end also refer to full-length residue numbering
    d_crop = t.top.select(f"resid {start-1} to {end-1}")
    t_crop = t.atom_slice(d_crop)

    write_pdb_with_bfactors(pdb_in, f"input/{gene}.pdb", t_crop, d_crop)

def crop_pae(pae_data, start, end):
    entry = pae_data[0] if isinstance(pae_data, list) else pae_data
    pae = entry["predicted_aligned_error"]

    i0 = start - 1
    i1 = end

    cropped = [row[i0:i1] for row in pae[i0:i1]]

    return [{
        "predicted_aligned_error": cropped,
        "max_predicted_aligned_error": entry["max_predicted_aligned_error"]}]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("uniprot_id", help="UniProt accession, e.g. P00698")
    parser.add_argument("start", type=int, help="First residue to keep, 1-based")
    parser.add_argument("end", type=int, help="Last residue to keep, inclusive")
    parser.add_argument("--tmd", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--orientation", nargs=3, type=float, default=[0, 0, 1])
    args = parser.parse_args()

    metadata = download_json(
        f"https://alphafold.ebi.ac.uk/api/prediction/{args.uniprot_id}")[0]

    gene = metadata.get("gene", args.uniprot_id)
    pae_url = metadata["paeDocUrl"]
    pdb_url = metadata["pdbUrl"]

    pae_data = download_json(pae_url)
    download_pdb(metadata["pdbUrl"], f"input/{gene}_FL.pdb")
    
    align_pdb(gene, start=args.start, end=args.end, indexes=args.tmd, orientation=args.orientation)

    cropped_pae = crop_pae(pae_data, args.start, args.end)
    pae_out = f"input/{gene}.json"
    with open(pae_out, "w") as f:
        json.dump(cropped_pae, f)

if __name__ == "__main__":
    main()
