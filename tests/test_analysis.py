from pathlib import Path
from types import SimpleNamespace

import mdtraj as md
import numpy as np
import pandas as pd
import pytest

from calvados import analysis


def test_cmap_traj_normalizes_selected_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe = SimpleNamespace(trajectory=list(range(10)))
    domain0 = [object(), object()]
    domain1 = [object(), object(), object()]
    monkeypatch.setattr(
        analysis,
        "calc_cmap",
        lambda *_: np.ones((len(domain0), len(domain1))),
    )

    cmap = analysis.cmap_traj(
        universe,
        domain0,
        domain1,
        start=1,
        end=9,
        step=2,
    )

    np.testing.assert_array_equal(cmap, np.ones((2, 3)))


def test_calc_com_traj_uses_configured_input_and_step(tmp_path: Path) -> None:
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    topology.add_atom("CA", md.element.carbon, residue)

    xyz = np.arange(15, dtype=float).reshape(5, 1, 3) / 10
    trajectory = md.Trajectory(
        xyz,
        topology,
        unitcell_lengths=np.full((5, 3), 10.0),
        unitcell_angles=np.full((5, 3), 90.0),
    )
    trajectory[0].save_pdb(tmp_path / "top.pdb")
    trajectory.save_dcd(tmp_path / "centered.dcd")
    residues_file = tmp_path / "residues.csv"
    pd.DataFrame({"three": ["ALA"], "MW": [89.1]}).to_csv(
        residues_file,
        index=False,
    )

    slab = analysis.SlabAnalysis(
        "system",
        input_path=tmp_path,
        output_path=tmp_path,
        input_pdb="top.pdb",
        centered_dcd="centered.dcd",
        ref_chains=(0, 0),
        ref_name="protein",
    )
    slab.calc_com_traj(residues_file, step=2)

    com_traj = md.load_dcd(
        tmp_path / "system_com_traj.dcd",
        top=tmp_path / "system_com_top.pdb",
    )
    assert com_traj.n_frames == 3
