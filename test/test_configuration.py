##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import sympy
from ameli import Electron, Config


def test_configuration():
    conf = "f3d2"

    config = Config(conf)
    assert config.info.num_subshells == 2
    assert config.info.num_electrons == 5
    assert config.states.pool_size == 2 * ((2*3+1) + (2*2+1))
    assert config.num_states == 16380
    s = sympy.Rational(1,2)
    assert config.electrons((21,)) == (Electron(shell=1, l=2, ml=-1, s=s, ms=-s),)
