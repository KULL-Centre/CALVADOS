import numpy as np
import pytest

from calvados.components import RNA


def test_rna_angle_map_clips_cosine() -> None:
    """Nearly parallel vectors should produce a finite angle."""
    v1 = np.array(
        [-2.2074710981998043e128, 8.27921441558737e127, 1.5416303946906181e128]
    )
    v2 = np.array(
        [-2.207471098202292e128, 8.279214415593619e127, 1.541630394690393e128]
    )
    rna = RNA.__new__(RNA)
    rna.xinit = np.array([v1, np.zeros(3), np.zeros(3), np.zeros(3), v2])

    rna.calc_angmap()

    assert rna.angmap[0] == pytest.approx(0.0)
