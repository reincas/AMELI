##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare calculated matrix elements of perturbation Hamiltonians and
# squared reduced matrix elements of the unit tensor operator of GSA
# transitions with measured and calculated spectroscopic data from
# papers of W. T. Carnall stored in 'data_energy.py'.
#
# Note: Some corrections had to be applied to the published data to
# correct typos or incorrect results.
#
##########################################################################

import pytest
import numpy as np
import sympy as sp
from ameli import Matrix
from data_energy import SOURCES, RADIAL


def correct(name, data, corrections):
    """ Apply corrections with given name to the given dataset in place. """

    for key, index, old_value, new_value in corrections:
        if key != name:
            continue
        assert data[index] == sp.S(old_value), f"{data[index]} != {old_value}"
        data[index] = sp.S(new_value)


def normalise_radial(radial):
    """ Convert an alternative parameter set into the standard set of radial integrals "H1"-"H6". """

    # Get keys and initialize the normalised set.
    keys = list(radial.keys())
    assert "base" not in keys
    new_radial = {}

    # No conversion for standard parameters
    for key in list(keys):
        if key[:2] in ("H1", "H2", "H3", "H4", "H5", "H6"):
            new_radial[key] = radial[key]
            keys.remove(key)

    # Convert "F_k" and "P_k" to "H1/k" and "H6/k", respectively
    for key in list(keys):
        if key == "F_0":
            new_radial["H1/0"] = radial[key]
        elif key == "F_2":
            new_radial["H1/2"] = radial[key] * 225
        elif key == "F_4":
            new_radial["H1/4"] = radial[key] * 1089
        elif key == "F_6":
            new_radial["H1/6"] = radial[key] * 184041 / 25
        elif key == "P_2":
            new_radial["H6/2"] = radial[key] * 225
        elif key == "P_4":
            new_radial["H6/4"] = radial[key] * 1089
        elif key == "P_6":
            new_radial["H6/6"] = radial[key] * 184041 / 25
        else:
            continue
        keys.remove(key)

    # The "E^i" parameters need a linear transformation to "H1/k". "E^0" or "H1/0" can be used as an alternative
    # to "base" to shift the whole energy level spectrum
    if "E^1" in keys:

        # Build transformation matrix
        A = np.array([[1, 9 / 7, 0, 0],
                      [0, 1 / 42, 143 / 42, 11 / 42],
                      [0, 1 / 77, -130 / 77, 4 / 77],
                      [0, 1 / 462, 5 / 66, -1 / 66]])
        A[1, :] *= 225
        A[2, :] *= 1089
        A[3, :] *= 184041 / 25

        # With offset parameter
        if "E^0" in keys:
            F0, F2, F4, F6 = A @ np.array([radial[f"E^{i}"] for i in range(4)])
            for i in range(4):
                keys.remove(f"E^{i}")

        # Without offset parameter
        else:
            F0 = None
            A = A[1:, 1:]
            F2, F4, F6 = A @ np.array([radial[f"E^{i}"] for i in range(1, 4)])
            for i in range(1, 4):
                keys.remove(f"E^{i}")

        # Store the converted parameters
        if F0 is not None:
            new_radial[f"H1/0"] = F0
        new_radial[f"H1/2"] = F2
        new_radial[f"H1/4"] = F4
        new_radial[f"H1/6"] = F6

    # There should be no remaining parameters
    if len(keys) != 0:
        raise ValueError(f"Unknown radial integrals: {', '.join(keys)}!")

    # Return normalised set of radial integrals
    return new_radial


def calc_energies(config_name, radial):
    """ Calculate energy spectrum for given radial integrals. """

    # Normalize radial integrals to the standard "H1" to "H6"
    radial = normalise_radial(radial)
    assert "base" not in radial
    matrices = []
    for name, factor in radial.items():
        assert Matrix.exists(config_name, name, "SLJ")
        matrix = Matrix(config_name, name, "SLJ")
        matrices.append((matrix, factor))

    # Get J space indices and collapse the J spaces to the stretched states with M = -J
    states = matrices[0][0].states
    num_states = states.num_states

    # Irreducible representations for collapsed J spaces
    irepr = states.representation_lists(["S2", "L2", "J2", "num"])

    # Build the full perturbation Hamiltonian
    H = np.zeros((num_states, num_states), dtype=np.float64)
    for matrix, factor in matrices:
        array = matrix.info.array(np.float64)
        H += factor * array
    energies, eigenvectors = np.linalg.eigh(H)
    return energies, eigenvectors, irepr


def check_coulomb(data):
    # Data correction list
    corrections = data.get("correct", [])

    # Number of f electrons
    assert "num" in data
    num_electrons = data["num"]
    config_name = f"f{num_electrons}"

    # Get, correct, and sort energy levels
    if "energies" in data:
        assert isinstance(data["energies"], list)
        energies_ref = list(data["energies"])
        correct("energies", energies_ref, corrections)
    else:
        assert "energies/meas" in data
        assert isinstance(data["energies/meas"], list)
        assert "energies/meas-calc" in data
        assert isinstance(data["energies/meas-calc"], list)
        meas = list(data["energies/meas"])
        correct("energies/meas", meas, corrections)
        diff = list(data["energies/meas-calc"])
        correct("energies/meas-calc", diff, corrections)
        energies_ref = [m - d for m, d in zip(meas, diff)]
    indices = np.argsort(energies_ref)
    energies_ref = sorted(map(float, energies_ref))

    # Get and correct radial integrals
    assert "radial" in data
    assert isinstance(data["radial"], dict)
    radial = dict(data["radial"])
    correct("radial", radial, corrections)

    # Get, correct, and order values of the J quantum number
    assert "J" in data
    assert isinstance(data["J"], list)
    assert len(data["J"]) == len(energies_ref)
    J = [sp.S(value) for value in data["J"]]
    correct("J", J, corrections)
    J = [J[i] for i in indices]

    # Get, correct, and order the optional squared reduced matrix elements of unit tensor operators
    if "U2" in data:
        U = {}
        for key in ("U2", "U4", "U6"):
            assert key in data
            assert isinstance(data[key], list)
            assert len(data[key]) == len(energies_ref) - 1
            U[key] = list(data[key])
            correct(key, U[key], corrections)
            U[key] = [U[key][i - 1] for i in indices[1:]]
        U2 = U["U2"]
        U4 = U["U4"]
        U6 = U["U6"]
    else:
        U2 = U4 = U6 = None

    # Initialise success flag
    success = True

    # Use ground state energy from literature as reference
    energies_calc, intermediate, irepr = calc_energies(config_name, radial)
    energies_calc += energies_ref[0] - energies_calc[0]

    # State names
    names = [f'{irepr["S2"][i]}{irepr["L2"][i]}{irepr["num"][i]} {irepr["J2"][i]}' for i in range(len(energies_calc))]

    # Names and J values of intermediate states
    max_indices = np.argmax(np.abs(intermediate), axis=0)
    real_names = [names[i] for i in max_indices]
    real_J = [irepr["J2"][i] for i in max_indices]

    # Compare given energy levels with calculation and determine mean quadratic deviation
    diffs = []
    for i in range(len(energies_ref)):
        diff = energies_ref[i] - energies_calc[i]
        if real_J[i] != str(J[i]) or abs(diff) > 2.5:
            success = False
            ref = f"J={J[i]}: {energies_ref[i]:.0f}"
            calc = f"{real_names[i]}: {energies_calc[i]:.0f}"
            print(f"*** | level {i} | ref {ref} | calc {calc} | diff {diff:.1f} ***")
        diffs.append(diff)
    diffs = np.array(diffs)
    print(f"Energy differences: mean {diffs.mean():.3f}, max {diffs.max():.3f}")

    # Compare squared reduced matrix elements with calculation
    if U2 is not None:
        U_real = {}
        for k in (2, 4, 6):
            name = f"U/{k}"
            assert Matrix.exists(config_name, name, "SLJ", reduced=True)
            reduced = Matrix(config_name, name, "SLJ", reduced=True)
            reduced = reduced.info.array(np.float64)
            reduced = intermediate.T @ reduced @ intermediate
            reduced = np.power(reduced, 2)
            U_real[k] = reduced
        diffs = []
        for i in range(1, len(energies_ref)):
            du2 = abs(U_real[2][0, i] - U2[i - 1])
            du4 = abs(U_real[4][0, i] - U4[i - 1])
            du6 = abs(U_real[6][0, i] - U6[i - 1])
            if max(du2, du4, du6) >= 0.0005:
                success = False
                ref = f"J={J[i]}: {U2[i - 1]:.4f} {U4[i - 1]:.4f} {U6[i - 1]:.4f}"
                calc = f"{real_names[i]}: {U_real[2][0, i]:.4f} {U_real[4][0, i]:.4f} {U_real[6][0, i]:.4f}"
                print(f"*** | level {i} | ref {ref} | calc {calc} ***")
            diffs.extend([du2, du4, du6])
        diffs = np.array(diffs)
        print(f"Reduced differences: mean {diffs.mean():.6f}, max {diffs.max():.6f}")

    #assert success


# @pytest.mark.parametrize("name", [key for key in RADIAL if "blocking" not in RADIAL[key]])
def run_coulomb(name):
    # Select data set
    assert name in RADIAL
    data = RADIAL[name]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Run test
    check_coulomb(data)


if __name__ == "__main__":
    for key in RADIAL:
        if "blocking" in RADIAL[key]:
            continue
        if RADIAL[key]["num"] not in [1, 2, 3, 4, 10, 11, 12, 13]:
            print(f"=== Skipping {key}")
            continue
        print(f"=== Checking {key} ...")
        run_coulomb(key)
