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

import logging
import pytest
import re
import sympy as sp
from ameli import SPECTRAL, Transform
from data_states import SOURCES, STATES
from conftest import DEBUG

logging.getLogger(__name__)


def rational_range(start, stop):
    curr = sp.S(start)
    while curr < stop:
        yield curr
        curr += 1


@pytest.mark.parametrize("num_electrons", range(1, 14))
def test_states(num_electrons: int):

    # Skip large configurations for debugging
    if DEBUG and DEBUG < num_electrons < 14 - DEBUG:
        reason = "debugging"
        logging.info(f"Test skipped -> {reason}")
        pytest.skip(reason)

    # States determined by AMELI
    config_name = f"f{num_electrons}"
    transform = Transform(config_name)
    states = transform.col_states.names

    # Terms from literature
    num_ref = num_electrons if num_electrons <= 7 else 14 - num_electrons
    terms_ref = STATES[f"f{num_ref}"]
    assert terms_ref["source"] in SOURCES
    assert terms_ref["l"] == 3
    assert terms_ref["num"] == num_ref

    # Build literature states by appending J and M
    states_ref = []
    pattern = r"(\d)([A-Z])(\d*)\s+(\d)\s+\((\d{3})\)\((\d{2})\)([AB]?)"
    for term in terms_ref["states"]:
        match = re.match(pattern, term)
        if not match:
            print("**********", term)
        s = (sp.S(match[1]) - 1) / 2
        assert s.is_integer or (s.is_rational and s.denominator == 2), f"{s}"
        l = SPECTRAL.index(match[2].lower())
        for j in rational_range(abs(l - s), s + l + 1):
            for m in rational_range(-j, j + 1):
                m = ("+", "")[int(bool(m < 0))] + str(m)
                states_ref.append(f"{term} {j} {m}")

    # Sort states
    states = sorted(states)
    states_ref = sorted(states_ref)

    # Compare lists of states in order
    if states != states_ref:
        for state in sorted(set(states) - set(states_ref)):
            logging.error(f"*** AMELI-only state: |{state}>")
        for state in sorted(set(states_ref) - set(states)):
            logging.error(f"*** Ref-only state: |{state}>")
        logging.error(f"*** AMELI states: {len(states)}, Ref states: {len(states_ref)}")
    assert states == states_ref
    logging.info(f"Test states {config_name} finished -> success")
