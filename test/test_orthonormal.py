##########################################################################
# Copyright (c) 2025 Reinhard Caspary                                    #
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
from ameli import Transform


@pytest.mark.parametrize("dtype, num_electrons", list(product(("symbolic", "float64"), range(1, 3))))
def test_orthonormal(dtype: str, num_electrons: int):
    config_name = f"f{num_electrons}"
    transform = Transform(dtype, config_name)
    dtype = transform.dtype
    if dtype.is_symbolic:
        V = transform.matrix
        diff = (V.T * V - sp.eye(transform.num_states)).norm(sp.oo)
        success = diff == 0
        # diff = "" if success else f" result = {diff} abs"
    else:
        V = transform.matrix
        diff = np.max(np.abs(V.T @ V - np.eye(V.shape[0])) / (dtype.eps + dtype.eps * np.eye(V.shape[0])))
        success = diff < 100
        # diff = f" result = {diff:.0f} eps"
    # res = "passed" if success else "FAILED"
    # print(f"{config_name} ({dtype}) | Orthonormality test{diff}: {res}")
    assert success
