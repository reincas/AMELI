##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
from ameli import Matrix, lanthanide_matrices
from logger import log_console, log_file


if __name__ == "__main__":
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    loglevel = logging.DEBUG
    log_file("ameli_5.log", formatter, loglevel) ####### DEBUG
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    names = lanthanide_matrices()
    for num_electrons in (9,):#range(1, 14):
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM"):
                for name, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue
                    matrix = Matrix(dtype, config_name, name, space)
