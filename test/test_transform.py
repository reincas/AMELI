##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# LS transformation check for the chain of symmetry matrices.
#
##########################################################################

import pytest
from itertools import product
import numpy as np
import sympy as sp
from ameli import Matrix, get_transform


@pytest.mark.parametrize("dtype, num_electrons", list(product(("symbolic", "float64"), range(1, 5))))
def test_transform(dtype: str, num_electrons: int):
    """ LS transformation test. """

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = get_transform(dtype, config_name)
    dtype = transform.dtype

    # Test every tensor operator in the symmetry chain
    for i, name in enumerate(transform.col_states.tensor_chain):
        if name in ("sen", "num", "tau"):
            continue

        # Operator matrix and eigenvalues
        matrix = Matrix(dtype, config_name, name, "Product").matrix
        eigenvalues = transform.col_states.eigenvalue_lists()[name]

        # Symbolic test
        if dtype.is_symbolic:
            V = transform.matrix
            D = sp.diag(*eigenvalues)
            diff = (V.T * matrix * V - D).norm(sp.oo)
            success = diff == 0

        # Floating point test
        else:
            V = transform.matrix
            D = np.diag([dtype.dtype(sp.S(x)) for x in eigenvalues])
            diff = np.max(np.abs(V.T @ matrix @ V - D) / (dtype.eps + dtype.eps * np.abs(D)))
            success = diff < 1000

        # Result
        assert success
