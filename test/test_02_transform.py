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

import logging
import pytest
import sympy as sp
from ameli import Matrix, Transform
from conftest import DEBUG

logging.getLogger(__name__)


@pytest.mark.parametrize("num_electrons", range(1, 14))
def test_transform(num_electrons: int):
    """ LS transformation test. """

    # Skip large configurations for debugging
    if DEBUG and DEBUG < num_electrons < 14 - DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = Transform(config_name)

    # Test every tensor operator in the symmetry chain
    for i, name in enumerate(transform.col_states.tensor_chain):
        if name in ("sen", "num", "tau"):
            continue

        # Operator matrix and eigenvalues
        matrix = Matrix(config_name, name, "Product").matrix
        eigenvalues = transform.col_states.eigenvalue_lists()[name]

        # Symbolic test
        V = transform.matrix
        D = sp.diag(*eigenvalues)
        diff = (V.T * matrix * V - D).norm(sp.oo)
        success = diff == 0

        # Test result
        assert success
        logging.info(f"Test transform {config_name}/{name} finished -> success")
