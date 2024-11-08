import numpy as np
import json
import os

import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import distances

# Functions

def make_waterbox(x_ag,gridsize=2.,buffer=5.):
    """ 
    Make water box grid.
    Box and gridsize are in Angstroms.
    Buffer is the extra water around protein in all dimensions [A].
    """
    waterbox = []

    mins = np.min(x_ag,axis=0)
    maxs = np.max(x_ag,axis=0)

    box_dims = maxs - mins + 2.*buffer

    xs = np.arange(0,box_dims[0]+gridsize*0.001,gridsize)
    ys = np.arange(0,box_dims[1]+gridsize*0.001,gridsize)
    zs = np.arange(0,box_dims[2]+gridsize*0.001,gridsize)

    for x in xs:
        for y in ys:
            for z in zs:
                waterbox.append([x,y,z])
    waterbox = np.array(waterbox)
    x_ag -= mins
    x_ag += buffer
    return x_ag, waterbox

def calc_cross_dmap(pos0,pos1):
    """ Calculate cross distance map between all positions """
    dmap = distances.distance_array(pos0,pos1)
    return dmap

def draw_vec():
    """ Random unit vector on a sphere """
    while True:
        vec = np.random.random(size=3) - 0.5
        length = np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
        if length <= 0.5: # avoid box corner bias
            vec /= length
            return vec

def calc_clashes(x_ag,x_wat,clashcutoff):
    """ Remove clashes between protein and water """
    x_wat_clean = []
    for x in x_wat:
        x = np.reshape(x,(1,-1))
        dmap = calc_cross_dmap(x,x_ag)
        if np.min(dmap) > clashcutoff:
            x_wat_clean.append(x[0])
    x_wat_clean = np.array(x_wat_clean)
    return x_wat_clean

def calc_depth_single_rotation(x_ag,x_wat_clean):
    """ Calculate depth for single rotation """
    depth = []
    for x in x_ag:
        x = np.reshape(x,(1,-1))
        dmap = calc_cross_dmap(x,x_wat_clean)
        mindist = np.min(dmap)
        depth.append(mindist)
    return depth

def pool_residues(depth, u):
    
    # Average value per residue

    # Average all the depth values for residues
    # NO GAPS IN SEQUENCE

    natoms = len(u.atoms)
    nres = len(u.residues)
    firstres = min(u.atoms.resids)

    depth_res_list = [[] for _ in range(nres)]
    depth_res_per_atom = np.zeros((natoms))
    depth_res = np.zeros((nres))

    # compile list of atom depths for each residue
    for idx, (res, dep) in enumerate(zip(ag.resids, depth)):
        depth_res_list[res-firstres].append(dep)
    
    # average list per residue
    for idx, dep in enumerate(depth_res_list):
        depth_res[idx] = np.mean(dep)

    # Subtract clashcutoff
    depth_res -= (clashcutoff / 10.)
 
    # assign average res value to each atom (for b-factor vis.)
    for idx, (res, dep) in enumerate(zip(u.atoms.resids, depth)):
        depth_res_per_atom[idx] = depth_res[res-firstres]

    return depth_res, depth_res_per_atom

def write_bfac(fpdb, depth_res_per_atom):
    # depth_norm = depth_res_per_atom / np.max(depth_res_per_atom) * 100.

    u.add_TopologyAttr('tempfactors')
    u.atoms.tempfactors = depth_res_per_atom

    u.atoms.write(f'{fpdb}_depth.pdb',) # depth in b-factor column of pdb

def calc_depth(
        fpdb, dmap,
        nrotations=10, clashcutoff=4, gridsize=2,
        write_pdb = True):
    """ 
    Calculate depth per residue for protein configuration
    (multiple rotations in a water box).
    Return depth in nm
    """

    u = mda.universe(fpdb)
    ag = u.atoms

    # Depth calculation

    depths = []

    for nrot in range(nrotations):
        print(f'Rotation: {nrot}')

        # rotate protein randomly
        ag.translate(-ag.center_of_mass())
        angle = np.random.random() * 360.
        direction = draw_vec()
        ag.rotateby(angle,direction)

        # read atom positions
        x_ag = ag.positions

        # make grid of waterbox
        x_ag, x_wat = make_waterbox(x_ag,gridsize=gridsize)

        # remove clashes between water and protein
        x_wat_clean = calc_clashes(x_ag,x_wat,clashcutoff)

        # calculate depth
        depth = calc_depth_single_rotation(x_ag,x_wat_clean) 
        depths.append(depth)

    depths = np.array(depths)
    depth = np.mean(depths,axis=0) / 10. # in nm

    depth_res, depth_res_per_atom = pool_residues(depth, u)
    if write_pdb:
        write_bfac(fpdb, depth_res_per_atom)

    return depth_res
