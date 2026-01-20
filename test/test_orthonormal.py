##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Orthonormality check for LS transformation matrices.
#
##########################################################################

import pytest
from itertools import product
import numpy as np
import sympy as sp
from ameli import get_transform


@pytest.mark.parametrize("dtype, num_electrons", list(product(("symbolic", "float64"), range(1, 5))))
def test_orthonormal(dtype: str, num_electrons: int):
    """ LS transformation orthonormality test. """

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = get_transform(dtype, config_name)
    dtype = transform.dtype

    # Symbolic transformation test
    if dtype.is_symbolic:
        V = transform.matrix
        diff = (V.T * V - sp.eye(transform.num_states)).norm(sp.oo)
        success = diff == 0

    # Floating point transformation test
    else:
        V = transform.matrix
        diff = np.max(np.abs(V.T @ V - np.eye(V.shape[0])) / (dtype.eps + dtype.eps * np.eye(V.shape[0])))
        success = diff < 100

    # Result
    assert success
