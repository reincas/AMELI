##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Exact symbolic LS transformation check for the chain of symmetry
# matrices.
#
##########################################################################

import pytest
import sympy as sp
from ameli import Matrix, Transform


@pytest.mark.parametrize("num_electrons", range(1, 4))
def test_transform(num_electrons: int):
    """ LS transformation test. """

    # Transformation object
    config_name = f"f{num_electrons}"
    dtype = "symbolic"
    transform = Transform(dtype, config_name)
    dtype = transform.dtype

    # Test every tensor operator in the symmetry chain
    for i, name in enumerate(transform.col_states.tensor_chain):
        if name in ("sen", "num", "tau"):
            continue

        # Operator matrix and eigenvalues
        matrix = Matrix(dtype, config_name, name, "Product").matrix
        eigenvalues = transform.col_states.eigenvalue_lists()[name]

        # Symbolic test
        V = transform.matrix
        D = sp.diag(*eigenvalues)
        diff = (V.T * matrix * V - D).norm(sp.oo)
        success = diff == 0

        # Result
        assert success
