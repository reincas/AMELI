##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the static dictionary CASIMIR, with three
# Casimir objects storing eigenvalues and irreducible representations
# of the Casimir operators for the special orthogonal (rotational)
# groups SO(5) and SO(7) and the special group G_2. Each Casimir object
# translates between SymPy eigenvalues and irreducible representation
# strings of the respective operators.
#
##########################################################################

import sympy as sp


class Casimir:
    """ Translation class for a Casimir operator. """

    # Dictionary with irreducible representations as keys and eigenvalues as values
    values = {}

    # Dictionary with eigenvalues as keys and irreducible representations as values
    keys = {}

    def key(self, value: sp.Expr) -> str:
        """ Return the irreducible representation string of the given SymPy eigenvalue. """

        if value not in self.keys:
            raise ValueError(f"Unknown value {value}!")
        return self.keys[value]

    def value(self, key: str) -> sp.Expr:
        """ Return the SymPy eigenvalue of the given irreducible representation string. """
        if key not in self.values:
            raise ValueError(f"Unknown key {key}!")
        return self.values[key]


class CasimirRot(Casimir):
    """ Translation class for the Casimir operator of the special orthogonal (rotational) group SO(2l+1)
    in 2l+1 dimensions, with l = 1, 2, or 3. """

    def __init__(self, l):
        """ Calculate the SymPy eigenvalues of all irreducible representation strings of the Casimir operator. """

        self.l = l
        assert 1 <= l <= 3

        self.values = {}
        self.keys = {}
        for key in self.elements():
            value = self.evaluate(key)
            self.values[key] = value
            self.keys[value] = key

    def evaluate(self, w: str) -> sp.Expr:
        """ Return the SymPy eigenvalue of the given irreducible representation string. """

        assert w.startswith("(") and w.endswith(")")
        w = tuple(map(int, w[1:-1]))
        assert self.l == len(w)
        value = 0
        for i, wi in enumerate(w):
            value += wi * (wi + 2 * self.l - 1 - 2 * i)
        return sp.S(value) / (2 * (2 * self.l - 1))

    def elements(self, values=None):
        """ Recursive generator of all irreducible representation strings of the Casimir operator. """

        if values is None:
            values = []
            max_num = self.l
        else:
            max_num = values[-1] + 1
        for num in range(max_num):
            result = values + [num]
            if len(values) < self.l - 1:
                yield from self.elements(result)
            else:
                yield "(" + "".join(map(str, result)) + ")"


class CasimirSpecial(Casimir):
    """ Translation class for the Casimir operator of the special orthogonal (rotational) group G_2. """

    def __init__(self):
        """ Calculate the SymPy eigenvalues of all irreducible representation strings of the Casimir operator. """

        self.values = {}
        self.keys = {}
        for key in self.elements():
            value = self.evaluate(key)
            self.values[key] = value
            self.keys[value] = key

    def evaluate(self, u: str) -> sp.Expr:
        """ Return the SymPy eigenvalue of the given irreducible representation string. """

        assert u.startswith("(") and u.endswith(")")
        u = tuple(map(int, u[1:-1]))
        assert len(u) == 2
        value = u[0] * u[0] + u[1] * u[1] + u[0] * u[1] + 5 * u[0] + 4 * u[1]
        return sp.S(value) / 12

    def elements(self):
        """ Recursive generator of all irreducible representation strings of the Casimir operator. """

        for i in range(5):
            for j in range(i + 1):
                u = (i, j)
                yield "(" + "".join(map(str, u)) + ")"


# Export three Casimir translator objects for the Casimir operators for the special orthogonal rotational groups
# SO(5) and SO(7) and the special group G_2
CASIMIR = {"SO7": CasimirRot(l=3), "SO5": CasimirRot(l=2), "G2": CasimirSpecial()}
