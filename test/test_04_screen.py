##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Screening test for the 3-electron matrices H4/1 (screened by e_0, e_1),
# H4/5 (screened by e_2), and H4/9 (identical zero) according to [34] and
# the 2nd order spin-orbit interaction H6/0 (screened by H2) according
# to [24].
#
##########################################################################

import logging
import pytest
import sympy
from ameli import Matrix
from conftest import DEBUG

logging.getLogger(__name__)

SOURCES = {
    "[24]": {
        "authors": ["B. R. Judd", "H. M. Crosswhite", "H. Crosswhite"],
        "title": "Intra-Atomic Magnetic Interactions for f Electrons",
        "journal": "Physical Review",
        "volume": "169",
        "number": 1,
        "pages": "130-138",
        "year": "1968",
        "doi": "https://doi.org/10.1103/PhysRev.169.130"
    },
    "[34]": {
        "authors": ["B. R. Judd"],
        "title": "Three-Particle Operators for Equivalent Electrons",
        "journal": "Physical Review",
        "volume": "141",
        "number": 1,
        "pages": "4-14",
        "year": "1966",
        "doi": "https://doi.org/10.1103/PhysRev.141.4"
    },
}


@pytest.mark.parametrize("num_electrons", range(3, 12))
def test_triple(num_electrons: int):

    # Skip large configurations for debugging
    if DEBUG and DEBUG < num_electrons < 14 - DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # Configuration
    config_name = f"f{num_electrons}"

    # Coulomb interaction matrices
    f0 = Matrix(config_name, "H1/0", "Product").matrix
    f2 = Matrix(config_name, "H1/2", "Product").matrix
    f4 = Matrix(config_name, "H1/4", "Product").matrix
    f6 = Matrix(config_name, "H1/6", "Product").matrix

    # Racah version of Coulomb interaction matrices
    e0 = f0
    e1 = 9 * f0 / 7 + 75 * f2 / 14 + 99 * f4 / 7 + 5577 * f6 / 350
    e2 = 10725 * f2 / 14 - 12870 * f4 / 7 + 5577 * f6 / 10

    # Matrix H4/1 is screened by e_0 and e_1
    name = "H4/1"
    t1 = Matrix(config_name, name, "Product").matrix
    t1_alt = (num_electrons - 2) * (e0 / 245 - e1 / 210) * sympy.sqrt(1155)
    assert t1 == t1_alt
    logging.info(f"Matrix {name} @ {config_name} is equivalent to e0,e1 -> success")

    # Matrix H4/5 is screened by e_2
    name = "H4/5"
    t5 = Matrix(config_name, name, "Product").matrix
    t5_alt = -(num_electrons - 2) / (14 * sympy.sqrt(4290)) * e2
    assert t5 == t5_alt
    logging.info(f"Matrix {name} @ {config_name} is equivalent to e2 -> success")

    # Matrix H4/9 is zero
    name = "H4/9"
    t9 = Matrix(config_name, name, "Product").matrix
    assert t9 == sympy.zeros(t9.rows, t9.cols)
    logging.info(f"Matrix {name} @ {config_name} is zero -> success")


@pytest.mark.parametrize("num_electrons", range(2, 13))
def test_magnetic(num_electrons: int):

    # Skip large configurations for debugging
    if DEBUG and DEBUG < num_electrons < 14 - DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # Configuration
    config_name = f"f{num_electrons}"

    # Spin-orbit interaction matrix
    z = Matrix(config_name, "H2", "Product").matrix

    # Matrix H6/0 is screened by H2
    name = "H6/0"
    p0 = Matrix(config_name, name, "Product").matrix
    p0_alt = -z * (num_electrons - 1) / 3
    assert p0 == p0_alt
    logging.info(f"Matrix {name} @ {config_name} is equivalent to H2 -> success")
