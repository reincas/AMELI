##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Quantitative floating point LS transformation check for the chain of
# symmetry matrices.
#
##########################################################################

import logging
import numpy as np
import sympy as sp
from ameli import Matrix, get_transform
from logger import log_console, log_file


def check_transform(dtype, num_electrons):
    """ Quantitative transformation check. """

    logger = logging.getLogger()

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = get_transform(dtype, config_name)
    dtype = transform.dtype
    assert not dtype.is_symbolic

    # Test every tensor operator in the symmetry chain
    for i, name in enumerate(transform.col_states.tensor_chain):
        if name in ("sen", "num", "tau"):
            continue

        # Operator matrix and eigenvalues
        matrix = Matrix(dtype, config_name, name, "Product").matrix
        eigenvalues = transform.eigenvalue_lists()[name]

        # Transformation test
        V = transform.matrix
        D = np.diag([dtype.dtype(sp.S(x)) for x in eigenvalues])
        diff = np.max(np.abs(V.T @ matrix @ V - D) / (dtype.eps + dtype.eps * np.abs(D)))
        success = diff < 1000

        # Result
        res = "passed" if success else "*** FAILED ***"
        head = f"{dtype.name} | {config_name:<3s}"
        if i != 0:
            head = " " * len(head)
        logger.info(f"{head} | {name} | Transformation test: {diff:.0f} eps, {V.shape[0]} states -> {res}")


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("check_transform.log", formatter, loglevel, delete=True)
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    for num_electrons in range(1, 5):
        check_transform("float64", num_electrons)
