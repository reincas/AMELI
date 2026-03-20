##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Exact symbolic comparison of 1st order spin-spin and spin-other-orbit
# (H5) and 2nd order spin-orbit (H6) matrix elements from the literature
# for the f2 and f12 configurations with results from the AMELI package.
#
# Note: We use equation (3) in [24] for the comparison. This equation
# contains a typo. The correct sign factor is (-1)^(S'+L'+J) with L'.
#
##########################################################################

import itertools
import logging
import pytest
import sympy as sp
from sympy.physics.wigner import wigner_6j
from ameli import Matrix
from data_magnetic import SOURCES, MAGNETIC
from conftest import DEBUG

logging.getLogger(__name__)


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

    # Skip large configurations for debugging
    if DEBUG and num_electrons != DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # Rank of double tensor operators
    assert "rank" in data
    rank = data["rank"]

    # Name of tensor operator
    assert "name" in data
    name = data["name"]

    # Reduced LS matrix elements
    assert "reduced" in data
    elements = data["reduced"]

    # Tensor operator matrix
    success = True
    matrix = Matrix(config_name, name, "SLJM")
    array = matrix.matrix

    # J space indices
    states = matrix.states
    j_spaces = states.state_spaces("J2")

    # Sanity check for homogeneous J spaces
    for i, j in j_spaces:
        assert len(set(array[i:j, i:j].diagonal().tolist()[0])) == 1
    assert array.is_symmetric()

    # Collapse the J spaces to the stretched states with M = -J
    indices = [i for i, j in j_spaces]
    array = array[indices, indices]

    # Eigenvalues for collapsed J spaces
    eigenvalues = states.eigenvalue_lists(["S2", "L2", "J2"])
    for name in eigenvalues:
        eigenvalues[name] = [(sp.sqrt(4 * eigenvalues[name][i] + 1) - 1) / 2 for i in indices]

    # Irreducible representations for collapsed J spaces
    irepr = states.representation_lists(["S2", "L2", "J2", "num"])
    for name in irepr:
        irepr[name] = [irepr[name][i] for i in indices]

    # Initialize set of involved J spaces and list of relative space signs
    spaces = set()
    phases = set()

    # Loop through all final states
    for i in range(array.shape[0]):
        term_a = f'{irepr["S2"][i]}{irepr["L2"][i]}{irepr["num"][i]}'
        state_a = f'{irepr["S2"][i]}{irepr["L2"][i]}{irepr["num"][i]}{irepr["J2"][i]}'

        # Loop through initial states up to the diagonal
        for j in range(i + 1):
            term_b = f'{irepr["S2"][j]}{irepr["L2"][j]}{irepr["num"][j]}'
            state_b = f'{irepr["S2"][j]}{irepr["L2"][j]}{irepr["num"][j]}{irepr["J2"][j]}'

            # Quantum numbers S, L, J of final and initial states
            Sa = eigenvalues["S2"][i]
            La = eigenvalues["L2"][i]
            Ja = eigenvalues["J2"][i]
            Sb = eigenvalues["S2"][j]
            Lb = eigenvalues["L2"][j]
            Jb = eigenvalues["J2"][j]

            # Test matrix elements for zero value
            is_zero = array[i, j] == 0

            # Matrix element must be zero for Ja != Jb
            if Ja != Jb:
                assert is_zero
                continue

            # Factor of Wigner 6-j symbol
            factor = wigner_6j(Sb, Lb, Ja, La, Sa, rank)
            if factor == 0:
                assert is_zero
                continue

            # Get respective reduced LS matrix element
            if f"{term_a} {term_b}" in elements:
                reduced = sp.S(elements[f"{term_a} {term_b}"])
                element = f"< {term_a} | {term_b} >"
            elif f"{term_b} {term_a}" in elements:
                reduced = sp.S(elements[f"{term_b} {term_a}"])
                element = f"< {term_b} | {term_a} >"
            else:
                if not is_zero:
                    success = False
                    logging.error(f"*** ERROR: unknown LS element < {term_a} | {term_b} > = {array[i,j]}!")
                continue

            # Calculate matrix element with sign
            # Note: [24] states -1 ^ ((Sb + La + Ja) % 2) for the sign, which is wrong!
            value = factor * reduced
            if (Sb + Lb + Ja) % 2 != 0:
                value = -value

            # Testing both matrix elements for same magnitude and same or opposite sign
            is_positive = value == array[i,j]
            is_negative = value = -array[i,j]

            # LS diagonal element, sign always +1
            if term_a == term_b:
                if not is_positive:
                    success = False
                    logging.error(f"*** ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")
                continue

            # Both LS states with same sign
            if is_positive:
                spaces.add(state_a)
                spaces.add(state_b)
                phases.add((state_a, state_b, 1))
                continue

            # Both LS states with opposite sign
            if is_negative:
                spaces.add(state_a)
                spaces.add(state_b)
                phases.add((state_a, state_b, -1))
                continue

            # Different magnitude of given and calculated matrix element
            success = False
            logging.error(f"*** ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")

    # Stop if one element test failed
    assert success

    # Try to find global phases of all involved J spaces matching the signs of the literature states
    spaces = tuple(spaces)
    success = False
    for signs in itertools.product((+1, -1), repeat=len(spaces)):
        if all(signs[spaces.index(a)] * signs[spaces.index(b)] == sign for a, b, sign in phases):
            success = True
            break

    # Test result
    assert success
    logging.info(f"Test magnetic {config_name}/{data_key} finished -> success")
