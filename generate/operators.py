##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the names of the angular matrices of all relevant
# tensor operators.
#
##########################################################################

from ameli.matrix import MatrixName


def matrix_names():
    """ Return a list of names of all available tensor operator matrices for lanthanide ions and the number of
    electron each one is acting on. """

    names = []
    names.extend([(f"U/{k},{q}", 1) for k in range(8) for q in range(-k, k + 1)])
    names.extend([(f"T/{k},{q}", 1) for k in range(2) for q in range(-k, k + 1)])
    names.extend([(f"UU/{k}", 1) for k in range(8)])
    names.extend([(f"TT/{k}", 1) for k in (0, 1)])
    names.extend([(f"UT/{k}", 1) for k in (0, 1)])
    names.extend([(f"C/{k},{q}", 1) for k in range(8) for q in range(-k, k + 1)])
    names.extend([(f"Hcf/{k},{q}", 1) for k in (0, 2, 4, 6) for q in range(0, k + 1)])
    names.extend([(f"Dcf/{k},{q}", 1) for k in (1, 3, 5, 7) for q in range(0, k + 1)])
    names.extend([(f"L/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"S/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"J/{q}", 1) for q in range(-1, 2)])
    names.append(("L2", 1))
    names.append(("S2", 1))
    names.append(("J2", 1))
    names.append(("LS", 1))
    names.extend([(f"H1/{k}", 2) for k in (0, 2, 4, 6)])
    names.append(("H2", 1))
    names.extend([(f"H3/{i}", 2) for i in (0, 1, 2)])
    names.extend([(f"H4/{c}", 3) for c in (1, 2, 3, 4, 5, 6, 7, 8, 9)])
    names.extend([(f"Hss/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"Hsoo/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H5/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H6/{k}", 2) for k in (0, 2, 4, 6)])
    return names


def matrix_args(num_electrons):
    """ Generator for the initialisation arguments (kwargs) of every available tensor operator matrix for a lanthanide
    ion with the given number of electrons. """

    # Loop through all types and names of tensor matrices
    reduced_names = set()
    config_name = f"f{num_electrons}"
    for state_space in ("Product", "SLJM", "SLJ"):
        for name, min_electrons in matrix_names():
            if num_electrons < min_electrons:
                continue

            # Crystal fields require full space
            if "cf" in name and state_space == "SLJ":
                continue

            # Yield initialisation arguments of a normal matrix
            kwargs = {
                "config_name": config_name,
                "name": name,
                "state_space": state_space,
                "reduced": False,
            }
            yield kwargs

            # Yield initialisation arguments of a matrix of reduced elements
            if state_space != "SLJ":
                continue

            name_data = MatrixName(name)
            if name_data.rank > 0:
                if "," in name:
                    reduced_name = name[:name.rfind(",")]
                else:
                    reduced_name = name[:name.rfind("/")]
            else:
                reduced_name = name
            if reduced_name in reduced_names:
                continue

            reduced_names.add(reduced_name)
            kwargs = {
                "config_name": config_name,
                "name": reduced_name,
                "state_space": state_space,
                "reduced": True,
            }
            yield kwargs
