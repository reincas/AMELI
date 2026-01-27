##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import time
from ameli import Matrix, lanthanide_matrices
from logger import log_console, log_file


def get_matrix(logger, dtype, config_name, name, space, reduced=False):
    matrix_str = f"Matrix({dtype}, {config_name}, {name}, {space}, reduced={reduced})"

    if Matrix.exists(dtype, config_name, name, space, reduced):
        logger.info(f"==== {matrix_str} exists.")
        return

    logger.info(f"==== Generate {matrix_str}.")
    t = time.time()
    matrix = Matrix(dtype, config_name, name, space, reduced)
    t = time.time() - t
    logger.info(f"==== Generating {matrix_str} took {t:.1f} seconds.")
    return matrix


def get_reduced(logger, dtype, config_name, name, space, matrix):
    if space != "SLJ":
        return

    if matrix is None:
        matrix = Matrix(dtype, config_name, name, space)

    k = matrix.rank
    if k > 0:
        if "," in name:
            name = name[:name.rfind(",")]
        else:
            name = name[:name.rfind("/")]

    if Matrix.exists(dtype, config_name, name, space, True):
        return

    return get_matrix(logger, dtype, config_name, name, space, True)


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("ameli.log", formatter, loglevel)  ####### DEBUG
    log_console(formatter, loglevel)
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    names = lanthanide_matrices()
    for num_electrons in [1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]:
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM", "SLJ"):
                for name, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue

                    matrix = get_matrix(logger, dtype, config_name, name, space)
                    reduced = get_reduced(logger, dtype, config_name, name, space, matrix)
