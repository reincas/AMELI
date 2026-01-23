##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import time
from ameli import Matrix, lanthanide_matrices
from logger import log_console, log_file


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("ameli.log", formatter, loglevel) ####### DEBUG
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    logger = logging.getLogger()
    names = lanthanide_matrices()
    for num_electrons in [9, 6, 8, 7]:#range(1, 14):
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM"):
                for name, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue
                    matrix_str = f"Matrix({dtype}, {config_name}, {name}, {space})"
                    if Matrix.exists(dtype, config_name, name, space):
                        logger.info(f"==== {matrix_str} exists.")
                    else:
                        logger.info(f"==== Generate {matrix_str}.")
                        t = time.time()
                        matrix = Matrix(dtype, config_name, name, space)
                        t = time.time() - t
                        logger.info(f"==== Generating {matrix_str} took {t:.1f} seconds.")
