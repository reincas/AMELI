##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import re


##########################################################################
# Description string formatting
##########################################################################

def desc_format(description, kwargs):
    """ Basic transformation for mata data description strings. """

    desc = str(description)
    desc = re.sub(r"\s*\n\s*", " ", desc)
    desc = re.sub(r"\s*<br>\s*", "\n", desc)
    desc = desc.strip()
    desc = desc.format(**kwargs)
    return desc


def lanthanide_matrices():
    """ Return a list of names of all available matrices for lanthanide ions and the number of electron each is
    acting on. """

    names = []
    names.extend([(f"U/{k},{q}", 1) for k in range(7) for q in range(-k, k + 1)])
    names.extend([(f"T/{k},{q}", 1) for k in range(2) for q in range(-k, k + 1)])
    names.extend([(f"UU/{k}", 1) for k in (0, 1, 2, 3, 4, 5, 6)])
    names.extend([(f"TT/{k}", 1) for k in (0, 1)])
    names.extend([(f"UT/{k}", 1) for k in (0, 1)])
    names.extend([(f"L/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"S/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"J/{q}", 1) for q in range(-1, 2)])
    names.append(("L2", 1))
    names.append(("S2", 1))
    names.append(("J2", 1))
    names.append(("LS", 1))
    names.extend([(f"H1/{k}", 2) for k in (2, 4, 6)])
    names.append(("H2", 1))
    names.extend([(f"H3/{i}", 2) for i in (0, 1, 2)])
    names.extend([(f"H4/{c}", 3) for c in (2, 3, 4, 6, 7, 8)])
    names.extend([(f"Hss/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"Hsoo/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H5/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H6/{k}", 2) for k in (0, 2, 4, 6)])
    return names


##########################################################################
# Import main classes
##########################################################################

from .config import SPECTRAL, Electron, Config
from .product import Product
from .unit import Unit
from .matrix import Matrix
from .transform import Transform
from .vault import container_vault
