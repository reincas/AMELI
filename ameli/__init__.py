##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import re


def desc_format(description, kwargs):
    """ Basic transformation for mata data description strings. """

    desc = str(description)
    desc = re.sub(r"\s*\n\s*", " ", desc)
    desc = re.sub(r"\s*<br>\s*", "\n", desc)
    desc = desc.strip()
    desc = desc.format(**kwargs)
    return desc


from .config import SPECTRAL, Electron, Config
from .product import Product
from .unit import Unit
from .matrix import Matrix
from .transform import Transform
