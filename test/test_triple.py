##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare 2st order Coulomb interaction matrix elements for three
# electrons (H4) from the literature for the f3 configuration with
# results from the Lanthanide package.
#
##########################################################################

import itertools
import pytest
import sympy as sp
from ameli import Matrix
from data_triple import SOURCES, TRIPLE


@pytest.mark.parametrize("data_key", TRIPLE.keys())
def test_triple(data_key):
    # Select data set
    assert data_key in TRIPLE
    data = TRIPLE[data_key]

    # Test source link
    assert "source" in data
    assert data["source"] in SOURCES

    # Number of f electrons
    assert "num" in data
    num_electrons = data["num"]
    config_name = f"f{num_electrons}"

    # Name of tensor operator
    assert "name" in data
    name = data["name"]

    # Common factor of all matrix elements
    assert "factor" in data
    factor = sp.S(data["factor"])

    # LS matrix elements
    assert "elements" in data
    elements = data["elements"]

    # Tensor operator matrix
    success = True
    matrix = Matrix("symbolic", config_name, name, "SLJM")
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
    eigenvalues = states.eigenvalue_lists(["J2"])
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
        term_a = f"{irepr["S2"][i]}{irepr["L2"][i]}{irepr["num"][i]}"
        state_a = f"{irepr["S2"][i]}{irepr["L2"][i]}{irepr["num"][i]}{irepr["J2"][i]}"

        # Loop through initial states up to the diagonal
        for j in range(i + 1):
            term_b = f"{irepr["S2"][j]}{irepr["L2"][j]}{irepr["num"][j]}"
            state_b = f"{irepr["S2"][j]}{irepr["L2"][j]}{irepr["num"][j]}{irepr["J2"][j]}"

            # Quantum number J of final and initial states
            Ja = eigenvalues["J2"][i]
            Jb = eigenvalues["J2"][j]

            # Test matrix elements for zero value
            is_zero = array[i, j] == 0

            # Matrix element must be zero for Ja != Jb
            if Ja != Jb:
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
                    print(f"ERROR: unknown LS element < {term_a} | {term_b} > = {array[i, j]}!")
                continue
            value = factor * reduced

            # Testing both matrix elements for same magnitude and same or opposite sign
            is_positive = value == array[i, j]
            is_negative = value = -array[i, j]

            # LS diagonal element, sign always +1
            if term_a == term_b:
                if not is_positive:
                    success = False
                    print(f"ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")
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
            print(f"ERROR: {name}[{i},{j}]: {element} {value} != {array[i, j]}")

    # Stop if one element test failed
    assert success

    # Try to find global phases of all involved J spaces matching the signs of the literature states
    spaces = tuple(spaces)
    success = False
    for signs in itertools.product((+1, -1), repeat=len(spaces)):
        if all(signs[spaces.index(a)] * signs[spaces.index(b)] == sign for a, b, sign in phases):
            success = True
            break
    assert success
