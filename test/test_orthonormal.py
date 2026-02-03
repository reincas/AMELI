##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Orthonormality check for exact symbolic LS transformation matrices.
#
##########################################################################

import pytest
import sympy as sp
from ameli import Transform


@pytest.mark.parametrize("num_electrons", range(1, 4))
def test_orthonormal(num_electrons: int):
    """ LS transformation orthonormality test. """

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = Transform(config_name)

    # Symbolic transformation test
    V = transform.matrix
    diff = (V.T * V - sp.eye(transform.num_states)).norm(sp.oo)
    success = diff == 0

    # Result
    assert success
