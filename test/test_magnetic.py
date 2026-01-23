##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare 1st order spin-spin and spin-other-orbit (H5) and 2nd order
# spin-orbit (H6) matrix elements from the literature for the f2 and f12
# configurations with results from the AMELI package.
#
# Note: We use equation (3) in [24] for the comparison. This equation
# contains a typo. The correct sign factor is (-1)^(S'+L'+J) with L'.
#
##########################################################################

import itertools
import pytest
import sympy as sp
from sympy.physics.wigner import wigner_6j
from ameli import SPECTRAL, Matrix
from data_magnetic import SOURCES, MAGNETIC


@pytest.mark.parametrize("data_key", MAGNETIC.keys())
def test_magnetic(data_key):
    # Select data set
    assert data_key in MAGNETIC
    data = MAGNETIC[data_key]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Number of f electrons
    assert "num" in data
    num_electrons = data["num"]
    config_name = f"f{num_electrons}"

    # Rank of double tensor operators
    assert "rank" in data
    rank = data["rank"]

    # Name of tensor operator
    assert "name" in data
    name = data["name"]

    # Reduced LS matrix elements
    assert "reduced" in data
    elements = data["reduced"]

    # Compare tensor operator matrix to the calculation
    success = True
    matrix = Matrix("symbolic", config_name, name, "SLJM")
    states = matrix.states
    array = matrix.matrix
    j_spaces = states.state_spaces("J2")

    # Sanity check for homogeneous J spaces
    for i, j in j_spaces:
        assert len(set(array[i:j, i:j].diagonal().tolist()[0])) == 1
    assert array.is_symmetric()

    # Collapse the M-space to the stretched states with M = -J
    indices = [i for i, j in j_spaces]
    array = array[indices, indices]

    eigenvalues = states.eigenvalue_lists(["S2", "L2", "J2"])
    for name in eigenvalues.keys():
        eigenvalues[name] = [(sp.sqrt(4 * eigenvalues[name][i] + 1) - 1) / 2 for i in indices]

    # Initialize set of SL states and list of phase signs
    sl_states = set()
    phases = []

    # Compare every matrix element
    for i in range(array.shape[0]):
        for j in range(i + 1):

            # Quantum numbers S, L, J of final and initial states
            Sa = eigenvalues["S2"][i]
            La = eigenvalues["L2"][i]
            Ja = eigenvalues["J2"][i]
            Sb = eigenvalues["S2"][j]
            Lb = eigenvalues["L2"][j]
            Jb = eigenvalues["J2"][j]

            # LS term names
            term_a = f"{2 * Sa + 1}{SPECTRAL[La].upper()}"
            term_b = f"{2 * Sb + 1}{SPECTRAL[Lb].upper()}"

            # Test matrix elements for zero value
            if matrix.dtype.is_symbolic:
                is_zero = array[i, j] == 0
            else:
                is_zero = abs(array[i, j]) < 1000 * matrix.dtype.eps

            # Matrix element must be zero for Ja != Jb
            if Ja != Jb:
                assert is_zero
                continue

            # Factor of Wigner 6-j symbol
            factor = wigner_6j(Sb, Lb, Ja, La, Sa, rank)
            if factor == 0:
                assert is_zero
                continue

            # Get respective reduced SL matrix element
            if f"{term_a} {term_b}" in elements:
                reduced = sp.S(elements[f"{term_a} {term_b}"])
                element = f"< {term_a} | {term_b} >"
            elif f"{term_b} {term_a}" in elements:
                reduced = sp.S(elements[f"{term_b} {term_a}"])
                element = f"< {term_b} | {term_a} >"
            else:
                if not is_zero:
                    success = False
                    print(f"ERROR: unknown LS element < {term_a} | {term_b} > = {array[i,j]}!")
                continue

            # Calculate matrix element with sign
            # Note: [24] states -1 ^ ((Sb + La + Ja) % 2) for the sign, which is wrong!
            value = factor * reduced
            if (Sb + Lb + Ja) % 2 != 0:
                value = -value

            if matrix.dtype.is_symbolic:
                is_positive = value == array[i,j]
                is_negative = value = -array[i,j]
            else:
                is_positive = abs(value - array[i,j]) < 1000 * matrix.dtype.eps
                is_negative = abs(value + array[i,j]) < 1000 * matrix.dtype.eps

            # SL diagonal element, sign always +1
            if term_a == term_b:
                if not is_positive:
                    success = False
                    print(f"ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")
                continue

            # Both SL states with same sign
            if is_positive:
                sl_states.add(term_a)
                sl_states.add(term_b)
                phases.append((term_a, term_b, 1))
                continue

            # Both SL states with opposite sign
            if is_negative:
                sl_states.add(term_a)
                sl_states.add(term_b)
                phases.append((term_a, term_b, -1))
                continue

            # Different magnitude of given and calculated matrix element
            success = False
            print(f"ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")

    # Stop if one element test failed
    assert success

    # Test for state phases which match the detected matrix element signs
    sl_states = tuple(sl_states)
    success = False
    for signs in itertools.product((+1, -1), repeat=len(sl_states)):
        if all(signs[sl_states.index(term_a)] * signs[sl_states.index(term_b)] == sign for term_a, term_b, sign in phases):
            success = True
            break
    assert success

