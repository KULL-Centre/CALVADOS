from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from calvados.cfg import Components, Config, Job
from calvados.inputmodels import ComponentInput, SimulationInput, validate_inputs


def test_config_uses_model_defaults_and_serializes_values(tmp_path: Path) -> None:
    config = Config(box=[8, 8, 8], temp=293.15, ionic=0.15, pH=7.0)

    assert config.config["box"] == [8.0, 8.0, 8.0]
    assert config.config["sysname"] == "default_simulation"

    config.write(tmp_path)
    written = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert written == config.config


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Config(
            box=[8, 8, 8],
            temp=293.15,
            ionic=0.15,
            pH=7.0,
            tempp=300,
        )


def test_components_validate_add_and_preserve_yaml_shape(tmp_path: Path) -> None:
    components = Components(
        fresidues=Path("residues.csv"),
        ffasta=Path("sequences.fasta"),
        nmol=1,
    )
    components.add(name="A", nmol=2)

    with pytest.raises(ValidationError):
        components.add(name="B", nmol=-1)

    components.write(tmp_path)
    written = yaml.safe_load((tmp_path / "components.yaml").read_text())

    assert written["defaults"]["fresidues"] == "residues.csv"
    assert written["defaults"]["nmol"] == 1
    assert written["system"] == {"A": {"nmol": 2}}


def test_job_uses_model_defaults_and_renders_template(tmp_path: Path) -> None:
    job = Job(template="robust.jinja", fbash=Path("/tmp/test.bashrc"))

    assert job.settings["template"] == "robust.jinja"
    assert job.settings["fbash"] == "/tmp/test.bashrc"
    assert job.settings["envname"] == "calvados"
    assert job.settings["batch_sys"] == "SLURM"
    assert job.settings["folder"].endswith("calvados/data/templates")

    config = Config(box=[8, 8, 8], temp=293.15, ionic=0.15, pH=7.0)
    components = Components(
        fresidues="residues.csv",
        ffasta="sequences.fasta",
    )
    components.add(name="A")
    config.write(tmp_path)
    components.write(tmp_path)
    job.write(tmp_path, config, components)

    submission = (tmp_path / "job.sh").read_text()
    assert "source /tmp/test.bashrc" in submission
    assert "conda activate calvados" in submission


def test_job_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Job(
            template="robust.jinja",
            fbash="/tmp/test.bashrc",
            queue="qgpu",
        )


def test_validate_inputs_resolves_component_defaults_and_overrides() -> None:
    config_model, component_models = validate_inputs(
        config={
            "box": [8, 8, 8],
            "temp": 293.15,
            "ionic": 0.15,
            "pH": 7.0,
        },
        components={
            "defaults": {
                "fresidues": "residues.csv",
                "ffasta": "sequences.fasta",
                "nmol": 1,
            },
            "system": {
                "A": {"nmol": 2},
                "B": {"molecule_type": "rna"},
            },
        },
    )

    assert isinstance(config_model, SimulationInput)
    assert all(
        isinstance(component, ComponentInput)
        for component in component_models.values()
    )
    assert component_models["A"].name == "A"
    assert component_models["A"].nmol == 2
    assert component_models["A"].molecule_type == "protein"
    assert component_models["B"].nmol == 1
    assert component_models["B"].molecule_type == "rna"
    assert component_models["B"].fresidues == "residues.csv"


def test_validate_inputs_rejects_malformed_component_sections() -> None:
    config = {
        "box": [8, 8, 8],
        "temp": 293.15,
        "ionic": 0.15,
        "pH": 7.0,
    }

    with pytest.raises(ValueError, match="Unknown components input sections"):
        validate_inputs(config, {"defaults": {}, "system": {}, "other": {}})

    with pytest.raises(TypeError, match="component system must be a mapping"):
        validate_inputs(config, {"defaults": {}, "system": []})


@pytest.mark.parametrize(
    ("component", "message"),
    [
        (
            {"name": "A", "fresidues": "residues.csv"},
            "ffasta must be provided when restraint is False",
        ),
        (
            {
                "name": "A",
                "fresidues": "residues.csv",
                "restraint": True,
                "fdomains": "domains.yaml",
            },
            "pdb_folder must be provided when restraint is True",
        ),
        (
            {
                "name": "A",
                "fresidues": "residues.csv",
                "restraint": True,
                "pdb_folder": "pdbs",
            },
            "fdomains must be provided for harmonic restraints",
        ),
        (
            {
                "name": "A",
                "molecule_type": "ptm_protein",
                "fresidues": "residues.csv",
                "restraint": True,
                "restraint_type": "go",
                "pdb_folder": "pdbs",
            },
            "ffasta must be provided for ptm_protein components",
        ),
    ],
)
def test_component_input_requires_compatible_input_sources(
    component: dict,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ComponentInput.model_validate(component)


def test_component_input_accepts_complete_restrained_inputs() -> None:
    harmonic = ComponentInput(
        name="harmonic",
        fresidues="residues.csv",
        restraint=True,
        pdb_folder="pdbs",
        fdomains="domains.yaml",
    )
    go = ComponentInput(
        name="go",
        fresidues="residues.csv",
        restraint=True,
        restraint_type="go",
        pdb_folder="pdbs",
    )

    assert harmonic.restraint_type == "harmonic"
    assert go.restraint_type == "go"


def simulation_input(**overrides) -> dict:
    values = {
        "box": [20, 20, 100],
        "temp": 293.15,
        "ionic": 0.15,
        "pH": 7.0,
    }
    values.update(overrides)
    return values


def test_simulation_input_resolves_one_duration() -> None:
    default = SimulationInput.model_validate(simulation_input())
    clock_limited = SimulationInput.model_validate(
        simulation_input(runtime=12)
    )

    assert default.steps == 100_000_000
    assert default.runtime is None
    assert clock_limited.steps is None
    assert clock_limited.runtime == 12

    with pytest.raises(
        ValidationError,
        match="Provide either steps or runtime, not both",
    ):
        SimulationInput.model_validate(simulation_input(steps=1000, runtime=12))


@pytest.mark.parametrize(
    "setting",
    [
        {"pressure_coupling": True},
        {"bilayer_eq": True},
        {"box_eq": True},
    ],
)
def test_simulation_input_requires_pressure(setting: dict) -> None:
    with pytest.raises(ValidationError, match="pressure must be provided"):
        SimulationInput.model_validate(simulation_input(**setting))


def test_simulation_input_validates_box_equilibration() -> None:
    with pytest.raises(
        ValidationError,
        match="box_eq and bilayer_eq cannot both be enabled",
    ):
        SimulationInput.model_validate(
            simulation_input(
                box_eq=True,
                bilayer_eq=True,
                pressure=(1, 1, 1),
            )
        )

    with pytest.raises(
        ValidationError,
        match="box_eq requires at least one scalable box direction",
    ):
        SimulationInput.model_validate(
            simulation_input(
                box_eq=True,
                pressure=(1, 1, 1),
                boxscaling_xyz=(False, False, False),
            )
        )


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({}, "slab_width must be provided"),
        ({"slab_width": 100}, "slab_width must be smaller"),
        (
            {"slab_width": 20, "slab_outer": 50},
            "slab_outer must be smaller than half",
        ),
        (
            {"slab_width": 20, "slab_outer": 9},
            "slab_outer must be greater than or equal",
        ),
    ],
)
def test_simulation_input_validates_slab_geometry(
    geometry: dict,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SimulationInput.model_validate(
            simulation_input(topol="slab", **geometry)
        )


def test_validate_inputs_requires_slab_outer_for_crowders() -> None:
    with pytest.raises(
        ValueError,
        match="slab_outer must be provided for slab systems containing crowders",
    ):
        validate_inputs(
            simulation_input(topol="slab", slab_width=20),
            {
                "defaults": {
                    "fresidues": "residues.csv",
                    "ffasta": "sequences.fasta",
                },
                "system": {
                    "crowder": {"molecule_type": "crowder"},
                },
            },
        )
