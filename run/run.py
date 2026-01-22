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
    log_file("ameli_h5.log", formatter, loglevel)
    log_console(formatter, loglevel)
    logging.getLogger().setLevel(loglevel)

    # names = lanthanide_matrices()
    names = []
    names.extend([(f"Hss/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"Hsoo/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H5/{k}", 2) for k in (0, 2, 4)])

    for num_electrons in range(1, 5):
        config_name = f"f{num_electrons}"
        for dtype in ("symbolic", "float64"):
            for space in ("Product", "SLJM"):
                for name, min_electrons in names:
                    if num_electrons < min_electrons:
                        continue
                    matrix = Matrix(dtype, config_name, name, space)
