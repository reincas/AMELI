##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

# import psutil
# def mem_info():
#     """ Virtual memory information for the detection of memory leaks. """
#     mem = psutil.virtual_memory()
#     total = mem.total / 1024 ** 3
#     used = mem.percent
#     avail = mem.available / 1024 ** 3
#     line = f"Virtual memory: {used:.1f} % used, {avail:.2f} / {total:.2f} GB available"
#     assert used < 85.0, f"*** EMERGENCY STOP: {line} ***"
#     return line
# print(f"Starting program --- {mem_info()}")

import logging
import time
import sys
from ameli import Matrix, lanthanide_matrices
from ameli.matrix import MatrixName
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


def get_reduced(logger, dtype, config_name, name, space):
    if space != "SLJ":
        return

    name_data = MatrixName(name)
    if name_data.rank > 0:
        if "," in name:
            name = name[:name.rfind(",")]
        else:
            name = name[:name.rfind("/")]

    if Matrix.exists(dtype, config_name, name, space, True):
        return

    return get_matrix(logger, dtype, config_name, name, space, True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nums = [int(sys.argv[1])]
        file = f"ameli-{nums[0]}.log"
    else:
        nums = [1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]
        file = f"ameli.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file(file, formatter, loglevel)
    log_console(formatter, loglevel)
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    logger.info(f"Prepare lanthanides: {nums}")

    names = lanthanide_matrices()
    for num_electrons in nums:
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM", "SLJ"):
                for name, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue

                    matrix = get_matrix(logger, dtype, config_name, name, space)
                    reduced = get_reduced(logger, dtype, config_name, name, space)
