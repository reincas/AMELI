##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Some matrices for almost closed shells.
#
##########################################################################

import logging
from ameli import Matrix, lanthanide_matrices


if __name__ == "__main__":
    logger = logging.getLogger()
    names = lanthanide_matrices()
    for num_electrons in [12, 13]:
        config_name = f"f{num_electrons}"
        dtype = "symbolic"
        space = "SLJM"
        for name, min_electrons in names:
            if num_electrons < min_electrons:
                continue
            if 14 - num_electrons < min_electrons:
                assert Matrix.exists(dtype, config_name, name, space)
                matrix = Matrix(dtype, config_name, name, space).matrix
                assert matrix.is_diagonal()
                values = set(matrix.diagonal().tolist()[0])

                matrix_str = f"Matrix({dtype}, {config_name}, {name}, {space})"
                values_str = ", ".join(map(str, values))
                result = ["*** FAILED ***", "success"][int(len(values) == 1)]
                print(f"{matrix_str}: {result} ({values_str})")
