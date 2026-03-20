##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Orthonormality check for exact symbolic LS transformation matrices.
#
##########################################################################

import logging
import pytest
import sympy as sp
from ameli import Transform
from conftest import DEBUG

logging.getLogger(__name__)


@pytest.mark.parametrize("num_electrons", range(1, 14))
def test_orthonormal(num_electrons: int):
    """ LS transformation orthonormality test. """

    # Skip large configurations for debugging
    if DEBUG and num_electrons != DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = Transform(config_name)

    # Symbolic transformation test
    V = transform.matrix
    diff = (V.T * V - sp.eye(transform.num_states)).norm(sp.oo)
    success = diff == 0

    # Test result
    assert success
    logging.info(f"Test orthonormal {config_name} finished -> success")
