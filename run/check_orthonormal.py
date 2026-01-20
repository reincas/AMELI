##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Quantitative orthonormality check for floating point LS transformation
# matrices.
#
##########################################################################

import logging
import numpy as np
from ameli import get_transform
from logger import log_console, log_file


def check_orthonormal(dtype, num_electrons):
    """ Quantitative orthonormality check. """

    logger = logging.getLogger()

    # Transformation object
    config_name = f"f{num_electrons}"
    transform = get_transform(dtype, config_name)
    dtype = transform.dtype
    assert not dtype.is_symbolic

    # Orthonormality check
    V = transform.matrix
    diff = np.max(np.abs(V.T @ V - np.eye(V.shape[0])) / (dtype.eps + dtype.eps * np.eye(V.shape[0])))
    success = diff < 1000

    # Result
    res = "passed" if success else "*** FAILED ***"
    logger.info(f"{dtype.name} | {config_name:<3s} | Orthonormality test: {diff:.0f} eps, {V.shape[0]} states -> {res}")


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("check_orthonormal.log", formatter, loglevel, delete=True)
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    for num_electrons in range(1, 5):
        check_orthonormal("float64", num_electrons)
