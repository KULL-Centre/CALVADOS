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
    components = Components(fresidues=Path("residues.csv"), nmol=1)
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
    components = Components(fresidues="residues.csv")
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
