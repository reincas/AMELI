##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Quantitative of floating point with symbolic matrices.
#
##########################################################################

import logging
import numpy as np
from ameli import Matrix, lanthanide_matrices
from logger import log_console, log_file


def check_matrix(dtype, num_electrons, space, names):
    """ Quantitative matrix check. """

    for i, name in enumerate(names):

        # Floating point matrix object
        config_name = f"f{num_electrons}"
        obj_num = Matrix(dtype, config_name, name, space)
        dtype = obj_num.dtype
        assert not dtype.is_symbolic

        # Symbolic matrix object
        obj_sym = Matrix("symbolic", config_name, name, space)

        # Matrix comparison test
        matrix_sym = dtype.array(obj_sym.matrix)
        matrix_num = obj_num.matrix
        diff = np.max(np.abs(matrix_sym - matrix_num)) / dtype.eps
        success = np.allclose(matrix_sym, matrix_num, atol=1000 * dtype.eps)

        # Result
        res = "passed" if success else "*** FAILED ***"
        head = f"{dtype.name} | {config_name:<3s} | {space}"
        if i != 0:
            head = " " * len(head)
        print(f"{head} | Matrix {name} test: {diff:.0f} eps -> {res}")


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("check_matrix.log", formatter, loglevel, delete=True)
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    all_names = lanthanide_matrices()
    dtype = "float64"
    space = "Product"
    for num_electrons in range(1, 5):
        names = [name for name, num in all_names if num <= num_electrons]
        check_matrix(dtype, num_electrons, space, names)
