##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Matrix, which represents the exact
# symbolic matrix of a angular spherical tensor operator in a given
# state space.
#
##########################################################################

import hashlib
import logging
import time

import sympy as sp

from . import desc_format, sym3j
from .sparse import SymMatrix
from .states import space_registry
from .config import ConfigInfo, Config
from .unit import Unit
from .vault import AMELI_VERSION, VersionError, Vault

__version__ = "1.0.1"
logger = logging.getLogger("matrix")


###########################################################################
# Unit spherical tensor operators
###########################################################################

def matrix_U(config, k: int, q: int):
    """ Return the matrix <|Uk_q|> of the q-component of the unit tensor operator of rank k acting in the orbital
    angular momentum space as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    assert 0 <= k <= 2 * l

    # One-electron matrix
    matrix = Unit(config.name, f"UT/{k},0,{k},{q}").matrix
    matrix *= sp.sqrt(2 * s + 1)

    # Return matrix
    return matrix


def matrix_T(config, k: int, q: int):
    """ Return the matrix <|Tk_q|> of the q-component of the unit tensor operator of rank k acting in the spin angular
    momentum space as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    assert 0 <= k <= 2 * s

    # One-electron matrix
    matrix = Unit(config.name, f"UT/0,{k},{k},{q}").matrix
    matrix *= sp.sqrt(2 * l + 1)

    # Return matrix
    return matrix


def matrix_UU(config, k: int):
    """ Return the matrix <|Uk*Uk|> of the squared unit tensor operator of rank k acting in the orbital angular
    momentum space as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1, "One sub-shell only!"

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    ident = sp.SparseMatrix.eye(config.num_states)
    assert 0 <= k <= 2 * l
    sign = (-1) ** k

    # One-electron matrix
    matrix = num * ident / (2 * l + 1)

    # Two-electron matrix
    if num > 1:
        factor = sign * 2 * (2 * s + 1) * sp.sqrt(2 * k + 1)
        unit = Unit(config.name, f"UTUT/{k},0,{k},{k},0,{k},0,0").matrix
        matrix += factor * unit

    # Return matrix
    return matrix


def matrix_TT(config, k: int):
    """ Return the matrix <|Sk*Sk|> of the squared unit tensor operator of rank k acting in the spin angular momentum
    space as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1, "One sub-shell only!"

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    ident = sp.SparseMatrix.eye(config.num_states)
    assert 0 <= k <= 2 * s
    sign = (-1) ** k

    # One-electron matrix
    matrix = num * ident / (2 * s + 1)

    # Two-electron matrix
    if num > 1:
        factor = sign * 2 * (2 * l + 1) * sp.sqrt(2 * k + 1)
        unit = Unit(config.name, f"UTUT/0,{k},{k},0,{k},{k},0,0").matrix
        matrix += factor * unit

    # Return matrix
    return matrix


def matrix_UT(config, k: int):
    """ Return the matrix <|Uk*Sk|> of the scalar product of the unit tensor operators of rank k acting in the orbital
    and the spin angular momentum space as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1, "One sub-shell only!"

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    assert 0 <= k <= 2 * min(s, l)
    sign = (-1) ** k

    # One-electron matrix
    factor = sign * sp.sqrt(2 * k + 1)
    unit = Unit(config.name, f"UT/{k},{k},0,0").matrix
    matrix = factor * unit

    # Two-electron matrix
    if num > 1:
        factor = sign * 2 * sp.sqrt((2 * s + 1) * (2 * l + 1) * (2 * k + 1))
        unit = Unit(config.name, f"UTUT/{k},0,{k},0,{k},{k},0,0").matrix
        matrix += factor * unit

    # Return matrix
    return matrix


###########################################################################
# Angular momentum spherical tensor operators
###########################################################################

def matrix_L(config, q: int):
    """ Return the matrix <|L_q|> of the q-component of the total orbital angular momentum operator as product state
    matrix for the given electron configuration. """

    l = config.states.electron_pool[0].l
    matrix = Matrix(config.name, f"U/1,{q}", "Product").matrix
    matrix *= sp.sqrt(l * (l + 1) * (2 * l + 1))
    return matrix


def matrix_S(config, q: int):
    """ Return the matrix <|S_q|> of the q-component of the total spin angular momentum operator as product state
    matrix for the given electron configuration. """

    s = config.states.electron_pool[0].s
    matrix = Matrix(config.name, f"T/1,{q}", "Product").matrix
    matrix *= sp.sqrt(s * (s + 1) * (2 * s + 1))
    return matrix


def matrix_J(config, q: int):
    """ Return the matrix <|J_q|> of the q-component of the total angular momentum operator as product state matrix for
    the given electron configuration. """

    matrix = Matrix(config.name, f"L/{q}", "Product").matrix
    matrix += Matrix(config.name, f"S/{q}", "Product").matrix
    return matrix


def matrix_SS(config):
    """ Return the matrix <|S*S|> of the squared total spin angular momentum operator as product state matrix for the
    given electron configuration. """

    s = config.states.electron_pool[0].s
    matrix = Matrix(config.name, "TT/1", "Product").matrix
    matrix *= s * (s + 1) * (2 * s + 1)
    return matrix


def matrix_LL(config):
    """ Return the matrix <|L*L|> of the squared total orbital angular momentum operator as product state matrix for
    the given electron configuration. """

    l = config.states.electron_pool[0].l
    matrix = Matrix(config.name, "UU/1", "Product").matrix
    matrix *= l * (l + 1) * (2 * l + 1)
    return matrix


def matrix_LS(config):
    """ Return the matrix <|L*S|> of the scalar product of the total orbital and spin angular momentum operators as
    product state matrix for the given electron configuration. """

    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    matrix = Matrix(config.name, "UT/1", "Product").matrix
    matrix *= sp.sqrt(s * (s + 1) * (2 * s + 1) * l * (l + 1) * (2 * l + 1))
    return matrix


def matrix_JJ(config):
    """ Return the matrix <|J*J|> of the squared total angular momentum operator as product state matrix for the given
    electron configuration. """

    matrix = Matrix(config.name, "LL", "Product").matrix
    matrix += 2 * Matrix(config.name, "LS", "Product").matrix
    matrix += Matrix(config.name, "SS", "Product").matrix
    return matrix


###########################################################################
# Casimirs spherical tensor operators
###########################################################################

def matrix_C2(config):
    """ Return the matrix <|C2(G2)|> of the Casimir operator of the special group G2 as product state matrix for the
    given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1, "One sub-shell only!"

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    assert l == 3
    num = config.info.num_electrons
    ident = sp.SparseMatrix.eye(config.num_states)

    # One-electron matrix
    factor = sp.S(0)
    for k in [1, 5]:
        factor += 2 * k + 1
    matrix = (num * factor / (2 * l + 1)) * ident

    # Two-electron matrix
    if num > 1:
        for k in [1, 5]:
            factor = (-1) ** k * 2 * (2 * s + 1) * (2 * k + 1) * sp.sqrt(2 * k + 1)
            unit = Unit(config.name, f"UTUT/{k},0,{k},{k},0,{k},0,0").matrix
            matrix += factor * unit

    # Return matrix
    matrix /= 4
    return matrix


def matrix_CR(config):
    """ Return the matrix <|C2(SO(2l+1))|> of the Casimir operator of the special orthogonal (rotational) group
    SO(2*l+1) as product state matrix for the given electron configuration. """

    # The current code works only for a single-shell configuration
    assert config.info.num_subshells == 1, "One sub-shell only!"

    # Parameters
    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    ident = sp.SparseMatrix.eye(config.num_states)

    # One-electron matrix
    factor = sp.S(0)
    for k in range(1, 2 * l, 2):
        factor += 2 * k + 1
    matrix = (num * factor / (2 * l + 1)) * ident

    # Two-electron matrix
    if num > 1:
        for k in range(1, 2 * l, 2):
            factor = (-1) ** k * 2 * (2 * s + 1) * (2 * k + 1) * sp.sqrt(2 * k + 1)
            unit = Unit(config.name, f"UTUT/{k},0,{k},{k},0,{k},0,0").matrix
            matrix += factor * unit

    # Return matrix
    matrix /= (2 * l - 1)
    return matrix


###########################################################################
# Perturbation Hamilton spherical tensor operators
###########################################################################

def matrix_H1(config, k: int):
    """ Return the matrix <|H1(k)|> of the Coulomb first order perturbation Hamiltonian with rank k as product state
    matrix for the given electron configuration. """

    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    assert num >= 2
    assert k % 2 == 0
    assert 0 <= k <= 2 * l
    matrix = Unit(config.name, f"UTUT/{k},0,{k},{k},0,{k},0,0").matrix
    matrix *= (-1) ** k * (2 * s + 1) * sp.sqrt(2 * k + 1)
    matrix *= ((-1) ** l * (2 * l + 1) * sym3j(l, k, l, 0, 0, 0)) ** 2
    return matrix


def matrix_H2(config):
    """ Return the matrix <|H2|> of the spin-orbit first order perturbation Hamiltonian as product state
    matrix for the given electron configuration. """

    s = config.states.electron_pool[0].s
    l = config.states.electron_pool[0].l
    k = 1
    matrix = Unit(config.name, f"UT/{k},{k},0,0").matrix
    matrix *= (-1) ** k * sp.sqrt(2 * k + 1)
    matrix *= sp.sqrt(s * (s + 1) * (2 * s + 1) * l * (l + 1) * (2 * l + 1))
    return matrix


# This data table is required for the calculation of the perturbation hamiltonian H4 of intra-configuration
# interactions which is based on effective three-electron scalar products
JUDD_TABLE = [[
    (2, 2, 2),
    "-sqrt(11 / 1134)", "sqrt(605 / 5292)", "sqrt(32761 / 889056)",
    "sqrt(3575 / 889056)", "-sqrt(17303 / 396900)", "-sqrt(1573 / 8232)",
    "sqrt(264407 / 823200)", "sqrt(21879 / 274400)", "-sqrt(46189 / 231525)",
], [
    (2, 2, 4),
    "sqrt(4 / 189)", "-sqrt(6760 / 43659)", "sqrt(33 / 1372)",
    "-sqrt(325 / 37044)", "sqrt(416 / 33075)", "-sqrt(15028 / 305613)",
    "sqrt(28717 / 2778300)", "-sqrt(37349 / 926100)", "-sqrt(8398 / 694575)",
], [
    (2, 4, 4),
    "sqrt(1 / 847)", "-sqrt(1805 / 391314)", "-sqrt(4 / 33957)",
    "-sqrt(54925 / 373527)", "-sqrt(117 / 296450)", "sqrt(4693 / 12326391)",
    "-sqrt(1273597 / 28014525)", "sqrt(849524 / 9338175)", "-sqrt(134368 / 3112725)",
], [
    (2, 4, 6),
    "sqrt(26 / 3267)", "-sqrt(4160 / 754677)", "-sqrt(13 / 264)",
    "sqrt(625 / 26136)", "sqrt(256 / 571725)", "sqrt(1568 / 107811)",
    "sqrt(841 / 1960200)", "-sqrt(17 / 653400)", "-sqrt(15827 / 245025)",
], [
    (4, 4, 4),
    "-sqrt(6877 / 139755)", "sqrt(55016 / 717409)", "sqrt(49972 / 622545)",
    "sqrt(92480 / 1369599)", "sqrt(178802 / 978285)", "-sqrt(297680 / 5021863)",
    "-sqrt(719104 / 2282665)", "-sqrt(73644 / 2282665)", "-sqrt(2584 / 18865)",
], [
    (4, 4, 6),
    "sqrt(117 / 1331)", "-sqrt(195 / 204974)", "sqrt(52 / 1089)",
    "sqrt(529 / 11979)", "-sqrt(2025 / 18634)", "-sqrt(49 / 395307)",
    "-sqrt(1369 / 35937)", "sqrt(68 / 11979)", "0",
], [
    (2, 6, 6),
    "sqrt(2275 / 19602)", "sqrt(1625 / 143748)", "sqrt(325 / 199584)",
    "sqrt(6889 / 2195424)", "71 / 198", "-sqrt(1 / 223608)",
    "sqrt(625 / 81312)", "sqrt(1377 / 27104)", "sqrt(323 / 22869)",
], [
    (4, 6, 6),
    "sqrt(12376 / 179685)", "sqrt(88400 / 1185921)", "-sqrt(442 / 12705)",
    "-sqrt(10880 / 251559)", "-sqrt(1088 / 179685)", "-sqrt(174080 / 8301447)",
    "-sqrt(8704 / 3773385)", "-sqrt(103058 / 1257795)", "-sqrt(19 / 31185)",
], [
    (6, 6, 6),
    "sqrt(4199 / 539055)", "sqrt(29393 / 790614)", "sqrt(205751 / 784080)",
    "-sqrt(79135 / 1724976)", "sqrt(2261 / 1078110)", "sqrt(79135 / 175692)",
    "sqrt(15827 / 319440)", "-sqrt(8379 / 106480)", "-sqrt(98 / 1485)",
]]


def matrix_H4(config, c: int):
    """ Return the matrix <|H4(c)|> of the effective Coulomb second order perturbation Hamiltonian with index c as
    product state matrix for the given electron configuration. """

    def judd_factor(i: int, c: int):
        """ Extract tensor ranks and factor from row i and column c of the Judd table. """

        ranks = JUDD_TABLE[i][0]
        factor = sp.factorial(len(ranks))
        for rank in set(ranks):
            factor /= sp.factorial(ranks.count(rank))
        factor *= sp.S(JUDD_TABLE[i][c])
        return ranks, factor

    # Jud table has 9 columns
    assert 1 <= c <= 9

    # The perturbation H4 is an effective three-electron operator and thus requires at least three electrons in
    # the configuration
    assert config.info.num_electrons >= 3

    # Calculate and return the operator matrix as linear combination of triple-scalar products
    l = config.states.electron_pool[0].l
    assert l == 3
    matrix = sp.SparseMatrix(config.num_states, config.num_states, {})
    for i in range(len(JUDD_TABLE)):
        (k1, k2, k3), factor = judd_factor(i, c)
        unit = Unit(config.name, f"UUU/{k1},{k2},{k3}").matrix
        matrix += factor * sp.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1)) * unit
    matrix *= 6
    return matrix


def matrix_Hss(config, k: int):
    """ Return the matrix <|Hss(k)|> of the spin-spin first order perturbation Hamiltonian with rank k as product state
    matrix for the given electron configuration. """

    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    assert num >= 2
    assert k % 2 == 0
    assert 0 <= k <= 2 * l - 2

    ck0 = (-1) ** l * (2 * l + 1) * sym3j(l, k, l, 0, 0, 0)
    ck2 = (-1) ** l * (2 * l + 1) * sym3j(l, k + 2, l, 0, 0, 0)

    factor = -12 * ck0 * ck2 * sp.sqrt((k + 1) * (k + 2) * (2 * k + 1) * (2 * k + 3) * (2 * k + 5))
    unit = Unit(config.name, f"UTUT/{k},1,{k + 1},{k + 2},1,{k + 1},0,0").matrix
    matrix = factor * unit

    return matrix


def matrix_Hsoo(config, k: int):
    """ Return the matrix <|Hsoo(k)|> of the spin-other-orbit first order perturbation Hamiltonian with rank k as
    product state matrix for the given electron configuration. """

    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    assert num >= 2
    assert k % 2 == 0
    assert 0 <= k <= 2 * l - 2

    ck0 = (-1) ** l * (2 * l + 1) * sym3j(l, k, l, 0, 0, 0)
    ck2 = (-1) ** l * (2 * l + 1) * sym3j(l, k + 2, l, 0, 0, 0)

    factor = ck0 ** 2 * sp.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 1))
    unit = Unit(config.name, f"UTUT/{k},0,{k},{k + 1},1,{k},0,0").matrix
    unit += 2 * Unit(config.name, f"UTUT/{k + 1},0,{k + 1},{k},1,{k + 1},0,0").matrix
    matrix = factor * unit
    factor = ck2 ** 2 * sp.sqrt((2 * l + k + 3) * (2 * l - k - 1) * (k + 2) * (2 * k + 5))
    unit = Unit(config.name, f"UTUT/{k + 2},0,{k + 2},{k + 1},1,{k + 2},0,0").matrix
    unit += 2 * Unit(config.name, f"UTUT/{k + 1},0,{k + 1},{k + 2},1,{k + 1},0,0").matrix
    matrix += factor * unit
    matrix *= 2 * sp.sqrt(3 * (2 * k + 3))
    return matrix


def matrix_H5(config, k: int):
    """ Return the matrix <|H5(k)|> of the spin-spin and spin-other-orbit first order perturbation Hamiltonian with
    rank k as product state matrix for the given electron configuration. """

    matrix_ss = Matrix(config.name, f"Hss/{k}", "Product").matrix
    matrix_soo = Matrix(config.name, f"Hsoo/{k}", "Product").matrix
    matrix = matrix_ss + matrix_soo
    return matrix


def matrix_H6(config, k: int):
    """ Return the matrix <|H6(k)|> of the effective electrostatic spin-orbit second order perturbation Hamiltonian
    with rank k as product state matrix for the given electron configuration. """

    l = config.states.electron_pool[0].l
    num = config.info.num_electrons
    assert num >= 2
    assert l == 3
    assert k % 2 == 0
    assert 0 <= k <= 2 * l

    factor = sp.sqrt((2 * l + k + 1) * (2 * l - k + 1) * k * (2 * k - 1))
    unit = Unit(config.name, f"UTUT/{k},0,{k},{k - 1},1,{k},0,0").matrix
    matrix = factor * unit

    if k < 2 * l:
        factor = -sp.sqrt((2 * l + k + 2) * (2 * l - k) * (k + 1) * (2 * k + 3))
        unit = Unit(config.name, f"UTUT/{k},0,{k},{k + 1},1,{k},0,0").matrix
        matrix += factor * unit

    ck = (-1) ** l * (2 * l + 1) * sym3j(l, k, l, 0, 0, 0)
    factor = -ck ** 2 / sp.sqrt(3 * (2 * k + 1))
    matrix *= factor
    return matrix


# Dictionary mapping name headers to matrix functions, tuples of operator parameters and a description string
MATRIX_INFO = [
    ("U", {"func": matrix_U, "keys": ("k", "q"),
           "desc": "Component q of the total unit tensor operator U^(k)_q of rank k in the orbital angular momentum space",
           "html_op": "$\\mathrm{{U}}^{{(k)}}_q$",
           "html_desc": "Component $q$ of the total unit tensor operator of rank $k$ in the orbital angular momentum space"}),
    ("T", {"func": matrix_T, "keys": ("k", "q"),
           "desc": "Component q of the total unit tensor operator T^(k)_q of rank k in the spin space",
           "html_op": "$\\mathrm{{T}}^{{(k)}}_q$",
           "html_desc": "Component $q$ of the total unit tensor operator of rank $k$ in the spin space"}),
    ("UU", {"func": matrix_UU, "keys": ("k",),
            "desc": "Squared total unit tensor operator [U^(k)]^2 of rank k in the orbital angular momentum space",
            "html_op": "$(\\mathrm{{U}}^{{(k)}}\\cdot\\mathrm{{U}}^{{(k)}})$",
            "html_desc": "Squared total unit tensor operator of rank $k$ in the orbital angular momentum space"}),
    ("TT", {"func": matrix_TT, "keys": ("k",),
            "desc": "Squared total unit tensor operator [T^(k)]^2 of rank k in the spin space",
            "html_op": "$(\\mathrm{{T}}^{{(k)}}\\cdot\\mathrm{{T}}^{{(k)}})$",
            "html_desc": "Squared total unit tensor operator of rank $k$ in the spin space"}),
    ("UT", {"func": matrix_UT, "keys": ("k",),
            "desc": "Scalar product of the total unit tensor operators U^(k)*T^(k) of rank k in the orbital and spin spaces",
            "html_op": "$(\\mathrm{{U}}^{{(k)}}\\cdot\\mathrm{{T}}^{{(k)}})$",
            "html_desc": "Scalar product of the total unit tensor operators of rank $k$ in the orbital and spin spaces"}),
    ("L", {"func": matrix_L, "keys": ("q",),
           "desc": "Component q of the total orbital angular momentum operator L_q",
           "html_op": "$\\mathrm{{L}}_q$",
           "html_desc": "Component $q$ of the total orbital angular momentum operator"}),
    ("S", {"func": matrix_S, "keys": ("q",),
           "desc": "Component q of the total spin angular momentum operator S_q",
           "html_op": "$\\mathrm{{S}}_q$",
           "html_desc": "Component $q$ of the total spin angular momentum operator"}),
    ("J", {"func": matrix_J, "keys": ("q",),
           "desc": "Component q of the total angular momentum operator J_q",
           "html_op": "$\\mathrm{{J}}_q$",
           "html_desc": "Component $q$ of the total angular momentum operator"}),
    ("LL", {"func": matrix_LL, "keys": (),
            "desc": "Squared total orbital angular momentum operator L^2",
            "html_op": "$(\\mathrm{{L}}\\cdot\\mathrm{{L}})$", "html_radial": "$\\alpha$",
            "html_desc": "Squared total orbital angular momentum operator"}),
    ("SS", {"func": matrix_SS, "keys": (),
            "desc": "Squared total spin angular momentum operator S^2",
            "html_op": "$(\\mathrm{{S}}\\cdot\\mathrm{{S}})$",
            "html_desc": "Squared total spin angular momentum operator"}),
    ("JJ", {"func": matrix_JJ, "keys": (),
            "desc": "Squared total angular momentum operator J^2",
            "html_op": "$(\\mathrm{{J}}\\cdot\\mathrm{{J}})$",
            "html_desc": "Squared total angular momentum operator"}),
    ("LS", {"func": matrix_LS, "keys": (),
            "desc": "Scalar product of the total orbital and spin angular momentum operators LS",
            "html_op": "$(\\mathrm{{L}}\\cdot\\mathrm{{S}})$",
            "html_desc": "Scalar product of the total orbital and spin angular momentum operators"}),
    ("C2", {"func": matrix_C2, "keys": (),
            "desc": "Casimir operator C_2(G_2) of the special group G_2",
            "html_op": "$\\mathrm{{C}}_2(G_2)$", "html_radial": "$\\beta$",
            "html_desc": "Casimir operator of the special group $G_2$"}),
    ("CR", {"func": matrix_CR, "keys": (),
            "desc": "Casimir operator C_2(SO(2l+1)) of the special orthogonal (rotational) group SO(2l+1) in 2l+1 dimensions",
            "html_op": "$\\mathrm{{C}}_2(SO(2l+1))$", "html_radial": "$\\gamma$",
            "html_desc": "Casimir operator of the special orthogonal (rotational) group in $2l+1$ dimensions"}),
    ("H1", {"func": matrix_H1, "keys": ("k",),
            "desc": "Coulomb first order perturbation Hamiltonian f_k of rank k",
            "html_op": "$\\mathrm{{f}}_k$", "html_radial": "$F^k$",
            "html_desc": "Coulomb first order perturbation Hamiltonian of rank $k$"}),
    ("H2", {"func": matrix_H2, "keys": (),
            "desc": "Spin-orbit first order perturbation Hamiltonian z",
            "html_op": "$\\mathrm{{z}}$", "html_radial": "$\\zeta$",
            "html_desc": "Spin-orbit first order perturbation Hamiltonian"}),
    ("H4", {"func": matrix_H4, "keys": ("c",),
            "desc": "Effective Coulomb second order perturbation Hamiltonian t_c with index c",
            "html_op": "$\\mathrm{{t}}_c$", "html_radial": "$T^c$",
            "html_desc": "Effective Coulomb second order perturbation Hamiltonian"}),
    ("Hss", {"func": matrix_Hss, "keys": ("k",),
             "desc": "Spin-spin first order perturbation Hamiltonian mss_k of rank k",
             "html_op": "$\\mathrm{{m}}_{{k,ss}}$",
             "html_desc": "Spin-spin first order perturbation Hamiltonian of rank $k$"}),
    ("Hsoo", {"func": matrix_Hsoo, "keys": ("k",),
              "desc": "Spin-other-orbit first order perturbation Hamiltonian msoo_k of rank k",
              "html_op": "$\\mathrm{{m}}_{{k,soo}}$",
              "html_desc": "Spin-other-orbit first order perturbation Hamiltonian of rank $k$"}),
    ("H5", {"func": matrix_H5, "keys": ("k",),
            "desc": "Spin-spin and spin-other-orbit first order perturbation Hamiltonian m_k of rank k",
            "html_op": "$\\mathrm{{m}}_k$", "html_radial": "$M^k$",
            "html_desc": "Spin-spin and spin-other-orbit first order perturbation Hamiltonian of rank $k$"}),
    ("H6", {"func": matrix_H6, "keys": ("k",),
            "desc": "Effective electrostatic spin-orbit second order perturbation Hamiltonian p_k of rank k",
            "html_op": "$\\mathrm{{p}}_k$", "html_radial": "$P^k$",
            "html_desc": "Effective electrostatic spin-orbit second order perturbation Hamiltonian of rank $k$"}),
]
MATRICES = dict(MATRIX_INFO)

# Translation of alternative names to canonical names
ALT_NAMES = {
    "Lz": "L/0",
    "Sz": "S/0",
    "Jz": "J/0",
    "L2": "LL",
    "S2": "SS",
    "J2": "JJ",
    "C7": "CR",
    "C5": "CR",
    "H3/0": "LL",
    "H3/1": "C2",
    "H3/2": "CR",
}


class MatrixName:
    """ Data class providing a couple of meta data derived from the components of a matrix name. """

    def __init__(self, name):
        """ Split the matrix name and initialize the data class. """

        # Name of the matrix
        if name in ALT_NAMES:
            name = ALT_NAMES[name]
        self.name = name

        # Split matrix name
        if "/" in self.name:
            self.head, args = self.name.split("/")
            self.args = tuple(map(int, args.split(",")))
        else:
            self.head = self.name
            self.args = ()
        info = MATRICES[self.head]
        self.func = info["func"]
        self.keys = info["keys"]
        self.tensor_desc = info["desc"]

        # Determine tensor rank
        if "q" not in self.keys:
            k = 0
        else:
            assert self.keys[-1] == "q"
            if "k" not in self.keys:
                k = 1
            else:
                assert self.keys[-2] == "k"
                k = self.args[self.keys.index("k")]
        self.rank = k

        # Missing parameters
        diff = len(self.keys) - len(self.args)
        assert diff >= 0
        self.missing = []
        if diff > 0:
            self.missing = self.keys[-diff:]

    def components(self):
        if self.missing == ("q",):
            sep = "/" if len(self.args) == 0 else ","
            names = [f"{self.name}{sep}{q}" for q in range(-self.rank, self.rank + 1)]
        elif not self.missing:
            names = [self.name]
        else:
            raise RuntimeError(f"Missing matrix parameters {self.missing} for matrix {self.name}!")
        return names


###########################################################################
# ProductMatrix class
###########################################################################

TITLE = "Spherical tensor operator matrix"

DESCRIPTION = """
This container contains the {reduced}matrix elements of a spherical tensor operator in the electron configuration 
{config_name}.
<br> {states_desc}
<br> {matrix_desc}
"""


class Matrix(Vault):
    """ Class representing the symbolic or floating point matrix of a spherical tensor operator in a given state
     space. The matrix object is available in the attribute 'matrix'. """

    def __init__(self, config_name, name, state_space, reduced=False):
        """ Initialize the spherical tensor operator matrix. """

        # Configuration string
        self.config_name = config_name

        # Translate alternative name to the canonical form
        if name in ALT_NAMES:
            name = ALT_NAMES[name]
        self.name = name

        # State space
        assert state_space in space_registry, f"Unknown state space '{state_space}'"
        assert not reduced or state_space == "SLJ"
        self.state_space = state_space
        self.reduced = bool(reduced)

        # Determine tensor rank
        self.rank = MatrixName(self.name).rank

        # Load or generate data container
        self.file = self.get_path(config_name, name, state_space, self.reduced)
        dc = self.load_container(self.file, __version__)

        # Extract UUID and code version from the container
        meta = dc["data/matrix.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Sanity check for data type, rank and reduced flag
        assert meta["dataType"] == "symbolic"
        assert self.rank == meta["tensorRank"]
        assert ("reduced" if self.reduced else "normal") == meta["elementType"]

        # Characteristics of the tensor operator
        assert meta["name"] == self.name
        self.tensor_parameters = meta["tensorParameters"]
        self.tensor_description = meta["tensorDescription"]

        # Extract electron configuration from the container
        self.config = ConfigInfo.from_meta(meta["config"])
        assert self.config.name == config_name

        # Extract states (Product or SLJM) from the container
        self.states = space_registry[self.state_space].from_meta(dc["data/states.hdf5"], meta["states"])
        assert self.states.state_space == state_space

        # Extract matrix
        self.info = SymMatrix.from_meta(dc["data/matrix.hdf5"], meta["matrix"])
        self.matrix = self.info.matrix
        assert self.info.row_space == self.state_space
        assert self.info.col_space == self.state_space

    def prepare_regular(self, config, space, hasher, dc=None):
        """ Return metadata dictionaries of states and matrix calculated from scratch. """

        # Decode matrix name
        name_data = MatrixName(self.name)

        # Calculate matrix elements
        matrix = name_data.func(config, *name_data.args)

        # Get states meta dictionaries
        space.load(self.config_name)
        states_dict, states_meta = space.as_meta(hasher)

        # Transform product space matrix to other coupling
        if space.matrix is not None:
            transform = space.matrix
            matrix = transform.T * matrix * transform

        # Get matrix metadata dictionaries
        if dc:
            state_matrix = SymMatrix.from_meta(dc["data/matrix.hdf5"], dc["data/matrix.json"]["matrix"])
        else:
            state_matrix = SymMatrix.from_matrix(self.state_space, self.state_space, matrix)
        matrix_dict, matrix_meta = state_matrix.as_meta(hasher)

        # Return metadata
        return states_dict, states_meta, matrix_dict, matrix_meta

    def prepare_slj(self, hasher, dc=None):
        """ Return metadata dictionaries of states and matrix derived from the collapsed respective SLJM matrix. """

        # Get SLJM parent matrix
        parent = Matrix(self.config_name, self.name, "SLJM")
        assert self.rank == parent.rank

        # Metadata dictionaries of states for collapsed J spaces
        states_dict, states_meta = parent.states.collapse_j().as_meta(hasher)
        indices = parent.states.indices_j()

        # Metadata dictionaries of matrix with collapsed J spaces
        if dc:
            state_matrix = SymMatrix.from_meta(dc["data/matrix.hdf5"], dc["data/matrix.json"]["matrix"])
            matrix_dict, matrix_meta = state_matrix.as_meta(hasher)
        else:
            matrix_dict, matrix_meta = parent.info.collapse(indices, "SLJM", "SLJ").as_meta(hasher)

        # Return metadata
        return states_dict, states_meta, matrix_dict, matrix_meta

    def prepare_reduced(self, hasher, dc=None):
        """ Return metadata dictionaries of states and reduced matrix derived from the respective SLJ matrices. """

        # Get component matrices of the tensor operator
        name_data = MatrixName(self.name)
        components = [Matrix(self.config_name, name, "SLJ") for name in name_data.components()]

        # Metadata dictionaries of states
        states = components[0].states
        states_dict, states_meta = states.as_meta(hasher)

        # Metadata dictionaries of matrix
        if dc:
            state_matrix = SymMatrix.from_meta(dc["data/matrix.hdf5"], dc["data/matrix.json"]["matrix"])
            matrix_dict, matrix_meta = state_matrix.as_meta(hasher)
        else:
            J = [sp.S(value) for value in states.representation_lists(["J2"])["J2"]]
            matrix = SymMatrix.reduced([matrix.info for matrix in components], J)
            matrix_dict, matrix_meta = matrix.as_meta(hasher)

        # Return metadata
        return states_dict, states_meta, matrix_dict, matrix_meta

    def generate_container(self, dc=None):
        """ Generate the matrix of the tensor operator and store it in a data container file. """

        # Get and use a logger object
        if dc:
            v = dc["meta.json"]["version"]
            logger.info(f"Update {self.config_name} tensor operator matrix {self.name} (version {v} -> {__version__})")
        else:
            logger.info(f"Generate {self.config_name} tensor operator matrix {self.name} (version {__version__})")
        t = time.time()

        # Initialize the data hasher
        hasher = hashlib.sha256()

        # Get electron configuration
        config = Config(self.config_name)
        config_meta = config.info.as_meta()

        # Get state space info
        space = space_registry[self.state_space]

        # Decode matrix name
        name_data = MatrixName(self.name)

        # Type of matrix elements
        element_type = {False: "normal", True: "reduced"}[self.reduced]

        # Metadata dictionaries of states and matrix
        if self.state_space == "SLJ" and not self.reduced:
            states_dict, states_meta, matrix_dict, matrix_meta = self.prepare_slj(hasher, dc)
        elif self.state_space == "SLJ" and self.reduced:
            states_dict, states_meta, matrix_dict, matrix_meta = self.prepare_reduced(hasher, dc)
        else:
            assert not self.reduced
            states_dict, states_meta, matrix_dict, matrix_meta = self.prepare_regular(config, space, hasher, dc)

        # Generate data hash
        data_hash = hasher.hexdigest()
        if dc and data_hash != dc["content.json"]["sha256Data"]:
            raise VersionError

        logger.debug(f" {self.config_name} | Finished tensor operator matrix {self.name}")

        # Prepare container description string
        kwargs = {
            "config_name": self.config_name,
            "reduced": "reduced " if self.reduced else "",
            "states": "states",
            "states_hdf5": "states.hdf5",
            "matrix_hdf5": "matrix.hdf5",
            "matrix": "matrix",
            "row_hdf5": "states.hdf5",
            "col_hdf5": "states.hdf5",
            "json": "matrix.json",
        }
        kwargs["states_desc"] = desc_format(space.states_desc, kwargs)
        kwargs["matrix_desc"] = desc_format(SymMatrix.meta_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Initialise container structure
        items = {
            "content.json": {
                "containerType": {"name": "ameliOperator"},
                "usedSoftware": [{"name": "AMELI", "version": AMELI_VERSION,
                                  "id": "https://github.com/reincas/ameli", "idType": "URL"}],
                "sha256Data": data_hash,
            },
            "meta.json": {
                "title": TITLE,
                "description": description,
                "license": "cc-by-sa-4.0",
                "version": __version__,
            },
            "data/matrix.json": {
                "version": __version__,
                "dataType": "symbolic",
                "name": self.name,
                "config": config_meta,
                "states": states_meta,
                "matrix": matrix_meta,
                "tensorParameters": dict(zip(name_data.keys, name_data.args)),
                "tensorDescription": name_data.tensor_desc,
                "tensorRank": self.rank,
                "elementType": element_type,
            },
            "data/states.hdf5": states_dict,
            "data/matrix.hdf5": matrix_dict,
        }

        # Create the data container and store it in a file
        self.write(self.file, items)
        t = time.time() - t
        logger.info(f"Stored {self.config_name} tensor operator matrix {self.name} ({t:.1f} seconds) -> {self.file}")

    @staticmethod
    def get_path(config_name, name, state_space, reduced):
        """ Return data container file name. """

        # Translate alternative name to the canonical form
        if name in ALT_NAMES:
            name = ALT_NAMES[name]

        # Sanity check for state space
        assert state_space in space_registry, f"Unknown state space '{state_space}'"
        if reduced:
            space = f"{state_space.lower()}_reduced"
        else:
            space = state_space.lower()

        # Return data container file name
        if "/" in name:
            head, args = name.split("/")
            args = "_".join(args.split(","))
            file = f"{config_name}/{space}/{head}_{args}.zdc"
        else:
            file = f"{config_name}/{space}/{name}.zdc"
        return file

    @staticmethod
    def exists(config_name, name, state_space, reduced=False):
        """ Return True if the matrix data container exists. """

        file = Matrix.get_path(config_name, name, state_space, reduced)
        return Vault.in_vault(file)
