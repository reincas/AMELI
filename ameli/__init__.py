##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import re
from functools import lru_cache
from sympy.physics.wigner import wigner_3j


def desc_format(description, kwargs):
    """ Basic transformation for metadata description strings. """

    desc = str(description)
    desc = re.sub(r"\s*\n\s*", " ", desc)
    desc = re.sub(r"\s*<br>\s*", "\n", desc)
    desc = desc.strip()
    desc = desc.format(**kwargs)
    return desc


@lru_cache(maxsize=100000)
def sym3j(j1, j2, j3, m1, m2, m3):
    """ Return the exact symbolic value of the Wigner 3j-symbol. The given arguments of the 3j-symbol must be integers
    or SymPy half-integer expressions. The results of this method are cached. """

    return wigner_3j(j1, j2, j3, m1, m2, m3)


from .config import SPECTRAL, Electron, Config
from .product import Product
from .unit import Unit
from .matrix import Matrix
from .transform import Transform
