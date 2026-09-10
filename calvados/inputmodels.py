from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

MoleculeType: TypeAlias = Literal[
    "protein",
    "rna",
    "lipid",
    "cooke_lipid",
    "crowder",
    "cyclic",
    "seastar",
    "ptm_protein",
]

InputPath: TypeAlias = PathLike[str] | str
RestraintChoices: TypeAlias = Literal["harmonic", "go"]
TopolType: TypeAlias = Literal[
    "single",
    "slab",
    "grid",
    "center",
    "shift_ref_bead",
    "random",
]

class ComponentInput(BaseModel):
    """Validate the definition of one molecular component.

    The model covers sequence and residue inputs, molecule counts, optional
    structure-based restraints, and parameters specific to RNA, branched, and
    post-translationally modified components. Unknown fields are rejected so
    misspelled configuration keys fail early.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    molecule_type: MoleculeType = 'protein'
    nmol: NonNegativeInt = 1
    charge_termini: Literal["none", "N", "C", "both"] = "both"
    alpha: float = 0.0
    kb: PositiveFloat = 8033.0
    fresidues: InputPath

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

    @model_validator(mode="after")
    def validate_component_inputs(self) -> Self:
        """Check that required component input sources were supplied."""
        if not self.restraint and self.ffasta is None:
            raise ValueError("ffasta must be provided when restraint is False")

        if self.molecule_type == "ptm_protein" and self.ffasta is None:
            raise ValueError("ffasta must be provided for ptm_protein components")

        if self.restraint and self.pdb_folder is None:
            raise ValueError("pdb_folder must be provided when restraint is True")

        if (
            self.restraint
            and self.restraint_type == "harmonic"
            and self.fdomains is None
        ):
            raise ValueError("fdomains must be provided for harmonic restraints")

        return self


class SimulationInput(BaseModel):
    """Validate global simulation, topology, and runtime settings.

    A simulation may be bounded by either ``steps`` or ``runtime``. The model
    also checks the combinations of slab geometry, pressure equilibration,
    restart behavior, custom restraints, and execution-platform options used
    when constructing an OpenMM simulation.
    """

    model_config = ConfigDict(extra="forbid")

    box: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    temp: PositiveFloat
    ionic: NonNegativeFloat
    pH: NonNegativeFloat

    sysname: str = "default_simulation"
    topol: TopolType = "center"

    fixed_lambda: NonNegativeFloat = 0.0
    eps_lj: NonNegativeFloat = 0.2
    cutoff_lj: NonNegativeFloat = 2.0
    cutoff_yu: NonNegativeFloat = 4.0

    steps: PositiveInt | None = None
    wfreq: PositiveInt = 100000
    platform: Literal["CPU","CUDA"] = "CPU"
    threads: PositiveInt = 1
    runtime: PositiveFloat | None = None
    restart: Literal["checkpoint","pdb","cif"] | None = "checkpoint"
    frestart: InputPath = "restart.chk"
    verbose: bool = False

    slab_eq: bool = False
    slab_eq_axis: tuple[bool, bool, bool] = (False, False, True)
    bilayer_eq: bool = False
    pressure_coupling: bool = False
    box_eq: bool = False
    pressure: tuple[float, float, float] | None = None
    boxscaling_xyz: tuple[bool, bool, bool] = (True, True, True)
    k_eq: NonNegativeFloat = 0.02
    steps_eq: NonNegativeInt = 1000
    ext_force: bool = False
    ext_force_expr: str = "step(d2-18)*d2; d2=periodicdistance(x, y, z, 0, 0, z)^2"

    friction_coeff: NonNegativeFloat = 0.01 
    slab_width: PositiveFloat | None = None
    slab_outer: PositiveFloat | None = None
    random_number_seed: int | None = None
    report_potential_energy: bool = False
    logfreq: PositiveInt = 1000000
    gpu_id: int = 0

    custom_restraints: bool = False
    custom_restraint_type: RestraintChoices = "harmonic"
    fcustom_restraints: InputPath = "custom_restraints.txt"

    ref_bead: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_simulation_duration(self) -> Self:
        """Require one simulation duration and populate the default step count."""
        if self.steps is not None and self.runtime is not None:
            raise ValueError("Provide either steps or runtime, not both")

        if self.steps is None and self.runtime is None:
            self.steps = 100_000_000

        return self

    @model_validator(mode="after")
    def validate_pressure_settings(self) -> Self:
        """Validate pressure-equilibration settings."""
        if self.box_eq and self.bilayer_eq:
            raise ValueError("box_eq and bilayer_eq cannot both be enabled")

        if self.box_eq and not any(self.boxscaling_xyz):
            raise ValueError("box_eq requires at least one scalable box direction")

        pressure_required = self.pressure_coupling or self.bilayer_eq or self.box_eq
        if pressure_required and self.pressure is None:
            raise ValueError(
                "pressure must be provided when pressure_coupling, "
                "bilayer_eq, or box_eq is enabled"
            )

        return self

    @model_validator(mode="after")
    def validate_external_force_settings(self) -> Self:
        """Prevent incompatible external restraint forces."""
        if self.slab_eq and self.ext_force:
            raise ValueError("slab_eq and ext_force cannot both be enabled")

        return self

    @model_validator(mode="after")
    def validate_slab_geometry(self) -> Self:
        """Ensure that the configured slab regions fit inside the box."""
        if self.topol != "slab":
            return self

        if self.slab_width is None:
            raise ValueError("slab_width must be provided for slab simulations")

        box_z = self.box[2]
        if self.slab_width >= box_z:
            raise ValueError("slab_width must be smaller than the box length in z")

        if self.slab_outer is not None:
            if self.slab_outer >= box_z / 2:
                raise ValueError(
                    "slab_outer must be smaller than half the box length in z"
                )
            if self.slab_outer < self.slab_width / 2:
                raise ValueError(
                    "slab_outer must be greater than or equal to slab_width / 2"
                )

        return self


class JobInput(BaseModel):
    """Validate settings used to render a cluster submission script.

    The selected Jinja template is populated with the environment setup file,
    Conda environment name, and either the SLURM or PBS batch-system choice.
    """

    model_config = ConfigDict(extra="forbid")

    template: InputPath = "robust.jinja"
    fbash: InputPath = Path("~/.bashrc")

    envname: str = 'calvados'
    batch_sys: Literal['SLURM', 'PBS'] = 'SLURM'


def validate_inputs(
    config: Mapping[str, Any] | SimulationInput,
    components: Mapping[str, Any],
) -> tuple[SimulationInput, dict[str, ComponentInput]]:
    """Validate simulation input and resolve every configured component."""
    config_model = SimulationInput.model_validate(config)

    if not isinstance(components, Mapping):
        raise TypeError("components input must be a mapping")

    unknown_sections = set(components) - {"defaults", "system"}
    if unknown_sections:
        raise ValueError(
            "Unknown components input sections: "
            f"{sorted(map(str, unknown_sections))}"
        )

    defaults = components.get("defaults", {})
    system = components.get("system", {})

    if not isinstance(defaults, Mapping):
        raise TypeError("component defaults must be a mapping")
    if not isinstance(system, Mapping):
        raise TypeError("component system must be a mapping")

    component_models: dict[str, ComponentInput] = {}
    for name, overrides in system.items():
        if not isinstance(name, str):
            raise TypeError("component names must be strings")
        if not isinstance(overrides, Mapping):
            raise TypeError(f"Component {name!r} must be a mapping")

        raw_component = {
            **defaults,
            **overrides,
            "name": name,
        }
        component_models[name] = ComponentInput.model_validate(raw_component)

    has_crowders = any(
        component.molecule_type == "crowder"
        for component in component_models.values()
    )

    molecule_types = {
        component.molecule_type for component in component_models.values()
    }
    if {"lipid", "cooke_lipid"}.issubset(molecule_types):
        raise ValueError(
            "lipid and cooke_lipid components cannot both be present"
        )

    for component in component_models.values():
        if component.molecule_type in ["lipid", "cooke_lipid", "crowder"]:
            if component.restraint:
                raise ValueError(
                    "Restraints cannot be used for lipids or crowders"
                )
        if (component.molecule_type == "rna") and (component.restraint_type == "go"):
            raise ValueError(
                "RNA restraints can only be harmonic."
            )

    if config_model.topol == "slab" and has_crowders:
        if config_model.slab_outer is None:
            raise ValueError(
                "slab_outer must be provided for slab systems containing crowders"
            )

    return config_model, component_models
