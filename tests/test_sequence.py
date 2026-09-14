from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calvados.sequence import (
    SeqFeatures,
    get_qs,
    patch_terminal_mws,
    patch_terminal_qs,
    seq_from_pdb,
)


def test_get_qs_standard_charges():
    qs, qs_abs = get_qs("KRHDEpA")

    expected = np.array([1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0])
    np.testing.assert_array_equal(qs, expected)
    np.testing.assert_array_equal(qs_abs, np.abs(expected))


def test_get_qs_custom_charges():
    residues = pd.DataFrame({"q": [0.25, -0.5]}, index=["A", "B"])

    qs, qs_abs = get_qs("ABA", residues=residues)

    expected = np.array([0.25, -0.5, 0.25])
    np.testing.assert_array_equal(qs, expected)
    np.testing.assert_array_equal(qs_abs, np.abs(expected))


def test_get_qs_flexible_histidine_charge():
    residues = pd.DataFrame({"q": [0.0, -0.25]}, index=["A", "H"])

    qs, qs_abs = get_qs("AH", residues=residues, flexhis=True, pH=6.0)

    expected = np.array([0.0, 0.5])
    np.testing.assert_array_equal(qs, expected)
    np.testing.assert_array_equal(qs_abs, expected)


def test_multichain_terminal_patching(tmp_path: Path):
    pdb = tmp_path / "multichain.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  LYS A   2       0.380   0.000   0.000  1.00  0.00           C\n"
        "TER       3      LYS A   2\n"
        "ATOM      4  CA  ASP B  10       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      5  CA  GLU B  11       1.380   0.000   0.000  1.00  0.00           C\n"
        "TER       6      GLU B  11\n"
        "END\n"
    )

    seq, n_termini, c_termini = seq_from_pdb(str(pdb))

    assert seq == "AKDE"
    assert n_termini == [0, 2]
    assert c_termini == [1, 3]
    np.testing.assert_array_equal(
        patch_terminal_qs(np.zeros(4), n_termini, c_termini),
        [1.0, -1.0, 1.0, -1.0],
    )
    np.testing.assert_array_equal(
        patch_terminal_mws(np.zeros(4), n_termini, c_termini),
        [2.0, 16.0, 2.0, 16.0],
    )


def test_seqfeatures():
    seq = "KAFD"
    residues = pd.DataFrame(
        {
            "q": [1.0, 0.0, 0.0, -1.0],
            "lambdas": [0.1, 0.2, 0.3, 0.4],
            "MW": [100.0, 110.0, 120.0, 130.0],
        },
        index=list(seq),
    )
    lambda_map = {
        (a, b): residues.lambdas[a] + residues.lambdas[b] for a in seq for b in seq
    }
    ah_integral_map = {(a, b): 2.0 for a in seq for b in seq}

    features = SeqFeatures(
        seq,
        residues=residues,
        lambda_map=lambda_map,
        ah_intgrl_map=ah_integral_map,
    )

    np.testing.assert_array_equal(features.qs, [1.0, 0.0, 0.0, -1.0])
    assert features.charge == pytest.approx(0.0)
    assert features.fcr == pytest.approx(0.5)
    assert features.ncpr == pytest.approx(0.0)
    assert features.scd == pytest.approx(-np.sqrt(3) / 4)
    assert features.faro == pytest.approx(0.25)
    assert features.mean_lambda == pytest.approx(0.25)
    assert features.shd == pytest.approx(13 / 24)
    assert features.mw == pytest.approx(460.0)
    assert features.ah_ij == pytest.approx(2.0)
