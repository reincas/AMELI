##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Compare calculated LS states with the states listed in the book from
# Nielson and Koster.
#
##########################################################################

import pytest
import re
from itertools import product
import sympy as sp
from ameli import SPECTRAL, get_transform

from data_states import SOURCES, STATES


def rational_range(start, stop):
    curr = sp.S(start)
    while curr < stop:
        yield curr
        curr += 1


@pytest.mark.parametrize("dtype, num_electrons", list(product(("symbolic", "float64"), range(1, 5))))
def test_states(dtype: str, num_electrons: int):

    # States determined by AMELI
    config_name = f"f{num_electrons}"
    transform = get_transform(dtype, config_name)
    states = transform.col_states.names

    # Terms from literature
    num_ref = num_electrons if num_electrons <= 7 else 14 - num_electrons
    terms_ref = STATES[f"f{num_ref}"]
    assert terms_ref["source"] in SOURCES
    assert terms_ref["l"] == 3
    assert terms_ref["num"] == num_ref

    # Build literature states by appending J and M
    states_ref = []
    pattern = r"(\d)([A-Z])(\d?)\s+(\d)\s+\((\d{3})\)\((\d{2})\)([AB]?)"
    for term in terms_ref["states"]:
        match = re.match(pattern, term)
        s = (sp.S(match[1]) - 1) / 2
        assert s.is_integer or (s.is_rational and s.denominator == 2), f"{s}"
        l = SPECTRAL.index(match[2].lower())
        for j in rational_range(abs(l - s), s + l + 1):
            for m in rational_range(-j, j + 1):
                m = ("+", "")[int(bool(m < 0))] + str(m)
                states_ref.append(f"{term} {j} {m}")

    # Compare lists of states in order
    if states != states_ref:
        print("AMELI:     ", sorted(set(states) - set(states_ref)))
        print("Literature:", sorted(set(states_ref) - set(states)))
    assert states == states_ref

