from typing import Literal
from pydantic import (
    Field,
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    ConfigDict,
    PositiveFloat,
    NonNegativeFloat,
)
from pathlib import Path

MoleculeType = Literal[
    "protein",
    "rna",
    "lipid",
    "cooke_lipid",
    "crowder",
    "cyclic",
    "seastar",
    "ptm_protein",
]

InputPath = Path | str
RestraintChoices = Literal["harmonic", "go"]
TopolType = Literal[
    "single",
    "slab",
    "grid",
    "center",
    "shift_ref_bead",
    "random",
]

class ComponentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    molecule_type: MoleculeType = 'protein'
    nmol: NonNegativeInt = 1
    charge_termini: Literal["none", "N", "C", "both"] = "both"
    alpha: float = 0.0
    kb: PositiveFloat = 8033.0

    fresidues: InputPath | None = None # maybe required?
    ffasta: InputPath | None = None

    restraint: bool = False
    restraint_type: RestraintChoices = "harmonic"
    cutoff_restr: NonNegativeFloat = 0.9
    pdb_folder: InputPath | None = None
    fdomains : InputPath | None = None
    k_harmonic: float = 700.0
    k_go: float = 15.0
    use_com: bool = True
    periodic: bool = False

    ext_restraint: bool = True

    colabfold: Literal[0,1,2] = 0
    bfac_shift: float = 0.8
    bfac_width: float = 50.0
    pae_shift: float = 0.3
    pae_width: float = 15.0

    rna_kb1: NonNegativeFloat = 8033.0
    rna_kb2: NonNegativeFloat = 8033.0
    rna_ka: NonNegativeFloat = 7.24
    rna_pa: NonNegativeFloat = 3.14
    rna_nb_sigma: NonNegativeFloat = 0.4
    rna_nb_scale: NonNegativeFloat = 15.0
    rna_nb_cutoff: NonNegativeFloat = 0.6

    n_ends: NonNegativeInt = 1

    ptm_name: str = "example_ptm"
    ptm_locations: list[PositiveInt] = Field(default_factory=list)


class SimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    temp: PositiveFloat
    ionic: NonNegativeFloat
    pH: NonNegativeFloat

    sysname: str = "default_simulation"
    topol: TopolType = "center"

    fixed_lambda: int = 0
    eps_lj: NonNegativeFloat = 0.2
    cutoff_lj: NonNegativeFloat = 2.0
    cutoff_yu: NonNegativeFloat = 4.0

    steps: PositiveInt = 100000000
    wfreq: PositiveInt = 100000
    platform: Literal["CPU","CUDA"] = "CPU"
    threads: PositiveInt = 1
    runtime: NonNegativeFloat = 0.0
    restart: Literal["checkpoint","pdb","cif"] | None = "checkpoint"
    frestart: InputPath = "restart.chk"
    verbose: bool = False

    slab_eq: bool = False
    slab_eq_axis: tuple[bool, bool, bool] = (False, False, True)
    bilayer_eq: bool = False
    pressure_coupling: bool = False
    box_eq: bool = False
    pressure: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boxscaling_xyz: tuple[bool, bool, bool] = (True, True, True)
    k_eq: NonNegativeFloat = 0.02
    steps_eq: NonNegativeInt = 1000
    ext_force: bool = False
    ext_force_expr: str = "step(d2-18)*d2; d2=periodicdistance(x, y, z, 0, 0, z)^2"

    friction_coeff: NonNegativeFloat = 0.01 
    slab_width: NonNegativeFloat = 100.0
    slab_outer: NonNegativeFloat = 40.0
    random_number_seed: int | None = None
    report_potential_energy: bool = False
    logfreq: PositiveInt = 1000000
    gpu_id: int = 0

    custom_restraints: bool = False
    custom_restraint_type: RestraintChoices = "harmonic"
    fcustom_restraints: InputPath = "custom_restraints.txt"

    ref_bead: NonNegativeInt = 0