##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Transform, which represents the exact
# symbolic orthonormal transformation matrix V from electron product
# states to LS coupling for a given many-electron configuration. The signs
# of the LS states are correctly correlated within the J spaces, which
# allows to build superpositions of Stark states with same J, but
# different M in intermediate coupling based on these LS states.
#
# The module also provides the class LS_States, which represents all
# LS states of a configuration.
#
##########################################################################

import hashlib
import logging
import time
import numpy as np
import sympy as sp

from . import desc_format, sym3j
from .states import space_registry, register_space, register_subspace
from .uintarray import decode_uint_array, encode_uint_array
from .sparse import SymMatrix
from .casimir import CASIMIR
from .config import SPECTRAL, ConfigInfo, Config
from .unit import Unit
from .matrix import Matrix
from .vault import AMELI_VERSION, VersionError, Vault

__version__ = "1.0.1"

logger = logging.getLogger("transform")
ATOL = 1e-12

###########################################################################
# LS term classification
###########################################################################

SYM_INFO = {
    "S2": {"matrix": "S2", "repr": lambda x: str(sp.sqrt(4 * x + 1)), "factor": 4,
           "desc": "Squared operator of the total spin, represented by the multiplicity 2S+1",
           "html_repr": "Multiplicity $2S+1$ derived from the eigenvalue $S(S+1)$",
           "html_op": "Squared operator of the total spin $(\\mathrm{S}\\cdot\\mathrm{S})$"},
    "C7": {"matrix": "C7", "repr": lambda x: CASIMIR["SO7"].key(x), "factor": 5,
           "desc": "Casimir operator of the special orthogonal group in 7 dimensions SO(7)",
           "html_repr": "3-tuple $W=(w_1w_2w_3)$",
           "html_op": "Casimir operator $\\mathrm{C}_2(SO(7))$ of the special orthogonal group in 7 dimensions"},
    "C5": {"matrix": "C5", "repr": lambda x: CASIMIR["SO5"].key(x), "factor": 3,
           "desc": "Casimir operator of the special orthogonal group in 5 dimensions SO(5)",
           "html_repr": "2-tuple $W=(w_1w_2)$",
           "html_op": "Casimir operator $\\mathrm{C}_2(SO(5))$ of the special orthogonal group in 5 dimensions"},
    "C2": {"matrix": "C2", "repr": lambda x: CASIMIR["G2"].key(x), "factor": 12,
           "desc": "Casimir operator of the special group G_2",
           "html_repr": "2-tuple $U=(u_1u_2)$",
           "html_op": "Casimir operator $\\mathrm{C}_2(G_2)$ of the special group $G_2$"},
    "L2": {"matrix": "L2", "repr": lambda x: SPECTRAL[(sp.sqrt(4 * x + 1) - 1) / 2].upper(), "factor": 4,
           "desc": "Squared operator of the total orbital angular momentum, represented by its spectral character",
           "html_repr": "Spectral character of $L$  derived from the eigenvalue $L(L+1)$",
           "html_op": "Squared operator of the total orbital angular momentum $(\\mathrm{L}\\cdot\\mathrm{L})$"},
    "J2": {"matrix": "J2", "repr": lambda x: str((sp.sqrt(4 * x + 1) - 1) / 2), "factor": 4,
           "desc": "Squared operator of the total angular momentum, represented by the quantum number J",
           "html_repr": "Quantum number $J$ derived from the eigenvalue $J(J+1)$",
           "html_op": "Squared operator of the total angular momentum $(\\mathrm{J}\\cdot\\mathrm{J})$"},
    "Jz": {"matrix": "Jz", "repr": lambda x: ("+", "")[int(bool(x < 0))] + str(x), "factor": 2,
           "desc": "Operator of the z component of the total angular momentum, represented by the quantum number M_J",
           "html_repr": "Eigenvalue $M_J$",
           "html_op": "Operator of the axial component of the total angular momentum $\\mathrm{J}_0$"},
    "sen": {"matrix": None, "repr": lambda x: str(x), "factor": None,
            "desc": "Seniority number",
            "html_repr": "Seniority number $\\nu$",
            "html_op": "Operators $(\\mathrm{S}\\cdot\\mathrm{S})$ and $\\mathrm{C}_2(SO(7))$"},
    "tau": {"matrix": None, "repr": lambda x: "" if x == 0 else chr(ord("A") + x - 1), "factor": None,
            "desc": "Index of different states with same set of quantum numbers",
            "html_repr": "Index $\\tau$ of different states with same set of quantum numbers",
            "html_op": ""},
    "num": {"matrix": None, "repr": lambda x: "" if x == 0 else str(x), "factor": None,
            "desc": "Index of different LS states with same L and S quantum numbers",
            "html_repr": "Index of different states with same quantum numbers $L$ and $S$",
            "html_op": ""},
}

SYM_CHAIN = {
    "default": {
        "chain": ["S2", "L2", "num", "J2", "Jz"],
        "sort": ["S2", "L2", "J2", "Jz"],
        "template_short": "{S2}{L2}{num}",
        "template_term": "{S2}{L2}{num}",
        "template_full": "{S2}{L2}{num} {J2} {Jz}",
    },
    "d(n)": {
        "chain": ["S2", "C5", "sen", "L2", "num", "J2", "Jz"],
        "sort": ["S2", "L2", "C5", "J2", "Jz"],
        "template_short": "{S2}{L2}{num}",
        "template_term": "{S2}{L2}{num} {sen} {C5}",
        "template_full": "{S2}{L2}{num} {sen} {C5} {J2} {Jz}",
    },
    "f(n)": {
        "chain": ["S2", "C7", "sen", "C2", "L2", "tau", "num", "J2", "Jz"],
        "sort": ["S2", "L2", "C7", "C2", "J2", "Jz"],
        "template_short": "{S2}{L2}{num}",
        "template_term": "{S2}{L2}{num} {sen} {C7}{C2}{tau}",
        "template_full": "{S2}{L2}{num} {sen} {C7}{C2}{tau} {J2} {Jz}",
    },
}


def config_key(config):
    """ Return the appropriate configuration key for the SYM_CHAIN dictionary. """

    key = "default"
    if config.info.num_subshells == 1:
        if config.states.electron_pool[0].l == 2:
            key = "d(n)"
        elif config.states.electron_pool[0].l == 3:
            key = "f(n)"
    return key


def str_terms(config, states: list, template=None) -> list:
    """ Return a list of term strings from the state list of eigenvalue dictionaries using the template with the
    given name from the dictionary SYM_CHAIN. Default is 'template_term'. Skip all duplicate term strings. """

    # Get the template string
    if template is None:
        template = "template_term"
    template = SYM_CHAIN[config_key(config)][template]

    # Build the list of unique term strings
    terms = []
    for state in states:
        state = {name: SYM_INFO[name]["repr"](value) for name, value in state.items()}
        term = template.format(**state)
        if term not in terms:
            terms.append(term)

    # Return the list of term strings
    return terms


def state_key(eigenvalues: dict, i: int, names: list) -> str:
    """ Return a string of space-separated eigenvalues of state i for the given list of operator name keys. """

    return " ".join([str(eigenvalues[name][i]) for name in names])


def build_tau(eigenvalues: dict) -> list:
    """ The lanthanide configurations f5 - f9 contain pairs of LS terms, which match in all eigenvalues in the
    chain of operators S2, C2(SO7), C2(G2), L2, J2, and Jz. These states get an artificial tau value of 1 or 2
    assigned ad-hoc represented by the labels "A" and "B". The tau value of unique states is 0. Return the list
    of tau values for all states. """

    # Get number of states with sanity check
    num_states = {len(values) for values in eigenvalues.values()}
    assert len(num_states) == 1, "Different numbers of eigenvalues!"
    num_states = num_states.pop()

    # Build a dictionary which maps every combination of eigenvalues to the respective state indices
    names = {}
    for i in range(num_states):
        if "C7" in eigenvalues:
            key = state_key(eigenvalues, i, ["S2", "C7", "C2", "L2", "J2", "Jz"])
        elif "C5" in eigenvalues:
            key = state_key(eigenvalues, i, ["S2", "C5", "L2", "J2", "Jz"])
        else:
            raise ValueError("Wrong configuration!")
        if key not in names:
            names[key] = [i]
        else:
            names[key].append(i)

    # Assign tau values 1 and 2 to matching pairs and 0 to states with a unique combination of eigenvalues
    tau_values = num_states * [0]
    for key in names:
        if len(names[key]) > 1:
            if len(names[key]) > 2:
                raise ValueError("More than 2 equal states!")
            for num, i in enumerate(names[key]):
                tau_values[i] = num + 1

    # Return the tau values of all states
    return tau_values


def build_num(eigenvalues: dict) -> list:
    """ Assign individual short-cut integer numbers to different LS terms which match in the quantum numbers L and S.
    The num value of unique LS-terms is 0. Return the list of num values for all states. """

    # Get number of states with sanity check
    num_states = {len(values) for values in eigenvalues.values()}
    assert len(num_states) == 1, "Different numbers of eigenvalues!"
    num_states = num_states.pop()

    # Build a two-level dictionary which maps every term to the respective state indices
    names = {}
    for i in range(num_states):

        # First level is the reference LS key
        ls_key = state_key(eigenvalues, i, ["S2", "L2"])
        if ls_key not in names:
            names[ls_key] = {}
            names[ls_key]["order"] = []

        # The second level key distinguishes LS terms with the same L and S values, but different term classification
        # otherwise. The list names[ls_key]["order"] is used to record the order in which every different other_key
        # appears in the loop.
        if "C7" in eigenvalues:
            other_key = state_key(eigenvalues, i, ["C7", "C2", "tau"])
        elif "C5" in eigenvalues:
            other_key = state_key(eigenvalues, i, ["C5", "tau"])
        else:
            raise ValueError("Wrong configuration!")
        if other_key not in names[ls_key]:
            names[ls_key][other_key] = [i]
            names[ls_key]["order"].append(other_key)
        else:
            names[ls_key][other_key].append(i)

    # Assign an integer num value each state
    num_values = num_states * [0]
    for ls_key in names:

        # The value of num stays 0 for states with unique LS value
        if len(names[ls_key]) <= 2:
            continue

        # For each other_key assign another num_value to every state with the same ls_key and other_key
        num_value = 1
        for other_key in names[ls_key]["order"]:
            for i in names[ls_key][other_key]:
                num_values[i] = num_value
            num_value += 1

    # Return the num values of all states
    return num_values


def build_sen(eigenvalues: dict) -> list:
    """ Return a list of the seniority numbers of all states. """

    # Get number of states with sanity check
    num_states = {len(values) for values in eigenvalues.values()}
    assert len(num_states) == 1, "Different numbers of eigenvalues!"
    num_states = num_states.pop()

    # Key of Casimir operator of the special orthogonal (rotational) group in 2l+1 dimensions SO(2l+1)
    if "C7" in eigenvalues:
        key = "C7"
    elif "C5" in eigenvalues:
        key = "C5"
    else:
        raise ValueError("Wrong configuration!")

    # Calculate the seniority number from the eigenvalues of the spin operator S2 and the Casimir operator
    sen_values = []
    for i in range(num_states):
        c = SYM_INFO[key]["repr"](eigenvalues[key][i]).count("2")
        S = (sp.sqrt(4 * eigenvalues["S2"][i] + 1) - 1) / 2
        sen_values.append(2 * c + 2 * S)

    # Return the seniority numbers of all states
    return sen_values


def classify_states(config, eigenvalues: dict):
    """ Add certain classification numbers to the dictionary of lists of eigenvalues for all states. """

    # Get the full list of tensor operator names as well as the names of classification numbers
    chain = SYM_CHAIN[config_key(config)]["chain"]

    # Add the required classification numbers as pseudo-eigenvalues
    if "tau" in chain:
        eigenvalues["tau"] = build_tau(eigenvalues)
    if "num" in chain:
        eigenvalues["num"] = build_num(eigenvalues)
    if "sen" in chain:
        eigenvalues["sen"] = build_sen(eigenvalues)

    # The set of eigenvalues and classification numbers must be complete now
    assert set(chain) == set(eigenvalues.keys())

    # Return the updated dictionary
    return eigenvalues


def get_states(config, eigenvalues: dict) -> list:
    """ Extract a dictionary of all eigenvalues and classification numbers for each state from the dictionary of
    lists of eigenvalues and return it as list. """

    # Get the full list of tensor operator names as well as the names of classification numbers
    chain = SYM_CHAIN[config_key(config)]["chain"]

    # Transpose and return the dictionary of state lists to a state list of eigenvalue dictionaries
    states = [{name: eigenvalues[name][i] for name in chain} for i in range(config.num_states)]
    return states


###########################################################################
# Global sign correction
###########################################################################

def reduced(index_a, index_b, operator, k, transform, J, Ma, Mb):
    """ Return a reduced matrix element <J||Q||J> of the given tensor operator Q of rank k by application of the
    Wigner-Eckart theorem on the matrix element <J,Ma|Q|J,Mb>. Return SymPy.nan if the Wigner-Eckart theorem
    can't be used for this matrix element. The function expects the product state matrices of the tensor coordinates
    in the dictionary 'operator' with the coordinate number q as key. The respective column vectors from the matrix
    'transform' are used to transform the tensor matrix element into the LS space. """

    # The required tensor coordinate is defined by the m-conditions of the 3j-symbol in the Wigner-Eckart theorem
    q = Ma - Mb
    if q < -k or q > k:
        return sp.nan

    # 3j-symbol with sign factor from the Wigner-Eckart theorem
    denominator = sym3j(J, k, J, -Ma, q, Mb)
    if denominator == 0:
        return sp.nan
    if (J - Ma) % 2:
        denominator = -denominator

    # Matrix element of the q-component of the tensor operator in LS coupling
    numerator = transform.col(index_a).T * operator[q] * transform.col(index_b)
    assert numerator.shape == (1, 1)
    numerator = numerator[0, 0]
    if numerator == 0:
        return sp.nan

    # Return valid non-zero reduced matrix element
    return numerator / denominator


def update_signs(slices, operator, k, transform, J, M, known, signs):
    """ Fix as many global signs in each of the J eigenspaces as possible using the Wigner-Eckart theorem on
    the given tensor operator of rank k. The eigenspaces are defined by the given list 'slices'. The dictionary
    'operator' contains the Sympy matrices of components of a tensor operator of rank k in the product space. Keys
    are the component number q and values are the matrices. Only negative components and q=0 are used here. The
    matrix 'transform' contains the LS eigenvectors as columns and the sequences J and M contain the respective
    quantum numbers of all states. The elements of the sequence 'known' are set True, if the sign of the respective
    eigenvector was determined and the element in 'signs' is set to 1 if the sign of the eigenvector must be changed,
    and to 0 otherwise. This function ignores all eigenvectors for which the respective element in 'known' is True. """

    # Loop over each J eigenspace
    for i, j in slices:

        # Single element space
        if J[i] == 0:
            known[i] = True
            continue

        # This eigenspace is already fixed completely
        if np.all(known[i:j]):
            continue

        # Take the reduced matrix of the stretched state M = J as reference, if it is not zero
        ref = j - 1
        if M[ref] != J[ref]:
            logger.error(f"Slice ({i},{j}): J={J[i:j]}")
            logger.error(f"Slice ({i},{j}): M={M[i:j]}")
        assert M[ref] == J[ref], f"M[{ref}]={M[ref]} != J[{ref}]={J[ref]}"
        diagonal = reduced(ref, ref, operator, k, transform, J[ref], M[ref], M[ref])
        if diagonal is sp.nan:
            continue
        known[ref] = True

        # Loop through all other eigenvectors
        for col in range(i, ref)[::-1]:

            # Sign was already fixed
            if known[col]:
                continue

            # Loop through all rows > col (positive coordinates of the tensor operator) limited by the tensor rank
            for row in range(col + 1, min(j, col + k + 1)):
                if not known[row]:
                    continue

                # Try to get a value for the reduced matrix element
                assert J[row] == J[col]
                element = reduced(row, col, operator, k, transform, J[row], M[row], M[col])
                if element is sp.nan:
                    continue

                # Apply sign of the row eigenvector
                if signs[row]:
                    element = -element

                # Fix sign of this eigenvector and proceed with the next one
                if element - diagonal == 0:
                    signs[col] = 0
                elif element + diagonal == 0:
                    signs[col] = 1
                else:
                    print(element, diagonal)
                    raise RuntimeError("Wigner-Eckart theorem failed!")
                known[col] = True
                break


def get_j_slices(config, eigenvalues, J, M):
    """ Return the list of slices of J eigenvalues. """

    num_states = config.num_states
    names = SYM_CHAIN[config_key(config)]["chain"][:-1]
    keys = [state_key(eigenvalues, i, names) for i in range(num_states)]
    slices = []
    i = 0
    for j in range(num_states + 1):
        if j == num_states or keys[j] != keys[i]:
            assert len(set(J[i:j])) == 1, keys[i]
            assert j - i == 2 * J[i] + 1, keys[i]
            assert M[i] == -J[i], keys[i]
            assert M[j - 1] == J[i], keys[i]
            slices.append((i, j))
            i = j
    return slices


def correct_signs(config, transform: sp.Matrix, eigenvalues: dict) -> sp.Matrix:
    """ Use the Wigner-Eckart theorem in the JM space to obtain consistent global signs of the eigenvectors in each
    J eigenspace. This allows to construct correct superpositions of M states. The matrix transform is returned with
    adjusted signs for all column vectors. """

    logger.info(f"Fixing global signs in the J eigenspaces {config.name}")

    # Number of states
    num_states = config.num_states
    assert num_states == transform.shape[0] == transform.shape[1]

    # No phase fixed yet
    known = np.zeros(num_states, dtype=bool)

    # Sign of each column vector
    signs = np.zeros(num_states, dtype=int)

    # Quantum numbers J and M of all states
    J = [(sp.sqrt(4 * x + 1) - 1) / 2 for x in eigenvalues["J2"]]
    M = eigenvalues["Jz"]

    # Slices of the J eigenspaces
    slices = get_j_slices(config, eigenvalues, J, M)

    # Get all unit tensor operator names
    unit_names = {"U": "UT/{k},0,{k},{q}", "T": "UT/0,{k},{k},{q}"}
    operators = []
    l = max(electron.l for electron in config.states.electron_pool)
    for k in range(1, 2 * l + 1):
        for op in ["U", "T"]:
            if op == "T" and k > 1:
                continue
            operators.append((k, f"{op}({k})", unit_names[op]))

    # Fix all signs in each J multiplet
    while not np.all(known):
        num_known = sum(known)
        for k, name, unit in operators:
            operator = {q: Unit(config.name, unit.format(k=k, q=q)).matrix for q in range(0, k + 1)}
            update_signs(slices, operator, k, transform, J, M, known, signs)
            logger.info(f"Global signs {name} in {config.name}: {sum(known)}/{len(known)} fixed")
            if np.all(known):
                break
        assert sum(known) > num_known

    # Make sure that all global signs are corrected
    if not np.all(known):
        eigenvalues = classify_states(config, eigenvalues)
        states = get_states(config, eigenvalues)
        terms = str_terms(config, states, template="template_full")
        for i in range(num_states):
            if not known[i]:
                print("Unresolved:", terms[i])
    assert np.all(known)

    # Apply the global signs
    for i in range(len(signs)):
        if signs[i]:
            transform[:, i] = -transform[:, i]

    # Return the corrected transformation matrix
    return transform


###########################################################################
# Simultaneous symbolic multi-matrix diagonalisation algorithm
###########################################################################

def duration(seconds: float) -> str:
    """ Convert a number of seconds to a H:MM:SS string. """

    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{seconds:02}"


def get_sym_matrices(config):
    """ Return three lists: the tensor keys of the symmetry chain, all symmetry matrices and all factors to obtain
    integer eigenvalues. """

    # Load or generate the symmetry operator matrix for every diagonalizing level
    chain = SYM_CHAIN[config_key(config)]["chain"]
    names = [name for name in chain if SYM_INFO[name]["matrix"]]
    matrices = [Matrix(config.name, SYM_INFO[name]["matrix"], "Product").matrix for name in names]

    # The product of the eigenvalues of each symmetry matrix with the respective factor results in integers
    factors = [SYM_INFO[name]["factor"] for name in names]

    # Return results
    return names, matrices, factors


def sort_states(config, transform, eigenvalues):
    """ Return sorted eigenvectors and eigenvalues. """

    # The process must have finished
    assert all(len(eigenvalues[name]) == config.num_states for name in eigenvalues)

    # Get sort order from the dictionary SYM_CHAIN
    sort_names = SYM_CHAIN[config_key(config)]["sort"]

    # Collect a list of eigenvalues in the sort order for each state
    elements = []
    for i in range(config.num_states):
        element = [eigenvalues[name][i] for name in sort_names]

        # Special case: Reverse the sorting order for the quantum number S
        if "S2" in sort_names:
            index = sort_names.index("S2")
            element[index] = -element[index]

        # Append the current state index to the current list of eigenvalues and append it to the collection
        element.append(i)
        elements.append(element)

    # Sort the collection of eigenvalues and retrieve the sort indices
    elements.sort()
    indices = [element[-1] for element in elements]
    assert len(indices) == config.num_states
    assert len(set(indices)) == config.num_states

    # Reorder the eigenvalues and eigenvectors according to the list of sort indices
    for name in eigenvalues:
        eigenvalues[name] = [eigenvalues[name][i] for i in indices]
    transform = transform[:, indices]

    # Return sorted eigenvectors and eigenvalues
    return transform, eigenvalues


class Result:
    """ Support class for the simultaneous symbolic multi-matrix diagonalisation algorithm. It provides the symmetry
    operator matrix for each level of the diagonalizing chain and accumulates the partial results received from the
    step-wise algorithm. """

    def __init__(self, config):

        # Store the Config object which is required for the calculation of the symmetry matrices
        self.config = config

        # Number of electron states of the given configuration
        self.num_states = self.config.num_states

        # Tensor keys of the symmetry chain, symmetry matrices and factors to obtain integer eigenvalues
        self.names, self.matrices, self.factors = get_sym_matrices(self.config)

        # The product of the eigenvalues of each symmetry matrix with the respective factor results in integers
        self.factors = [SYM_INFO[name]["factor"] for name in self.names]

        # Initialize an empty eigenvalue list for each diagonalization level
        self.eigenvalues = [[] for _ in range(len(self.names))]

        # Initialize an empty transformation matrix
        self.transform = sp.zeros(self.num_states, self.num_states)

        # Initialize the counter of valid eigenvectors
        self.finished = 0

        # Store the starting time for logging the elapsed and estimated residual time
        self.start = time.time()
        self.stop = None

    def add_values(self, level: int, value: sp.Expr, count: int):
        """ Add count equal eigenvalues of the symmetry operator matrix for the given diagonalizing level. """

        self.eigenvalues[level].extend([value] * count)

    def add_vectors(self, vectors: list):
        """ Add one or more common eigenvectors of all symmetry matrices. """

        # Add the given eigenvector(s) as columns to the transformation matrix
        for i, vector in enumerate(vectors):
            self.transform[:, self.finished + i] = vector

        # Update the eigenvector counter
        self.finished += len(vectors)

        # Store the final time when the last eigenvector was received
        if self.finished >= self.num_states:
            self.stop = time.time()

    def name(self, level: int) -> str:
        """ Return the name key of the symmetry operator for the given diagonalizing level. """

        return self.names[level]

    def matrix(self, level: int) -> sp.SparseMatrix:
        """ Return the symmetry operator matrix for the given diagonalizing level. """

        return self.matrices[level]

    def factor(self, level: int) -> int:
        """ Return the factor which turns every eigenvalue for the given diagonalizing level into an integer. """

        return self.factors[level]

    @property
    def max_level(self) -> int:
        """ Return the final diagonalizing level. """

        return len(self.names) - 1

    @property
    def rate(self) -> float:
        """ Return the average processing speed in eigenvectors per second. """

        return (time.time() - self.start) / self.finished

    @property
    def fraction(self) -> float:
        """ Return the processing state in percent of calculated eigenvectors. """

        return (self.finished / self.num_states) * 100

    @property
    def run_time(self) -> float:
        """ Return the current total run time in seconds. """

        return time.time() - self.start

    @property
    def remain_time(self) -> float:
        """ Return an estimate of the remaining processing time in seconds. """

        return (self.num_states - self.finished) * self.rate

    @property
    def total_time(self) -> float:
        """ Return the total run time in seconds after the process has finished. """

        assert self.stop is not None, "Process has not been finished!"
        return self.stop - self.start

    @property
    def status(self) -> str:
        """ Return a logging string with some statistics and the current run time and estimated remaining time. """

        c = self.config.name
        f = self.finished
        n = self.num_states
        p = self.fraction
        t = duration(self.run_time)
        if self.finished:
            e = duration(self.remain_time)
        else:
            e = "-:--:--"
        num = len(str(self.num_states))
        return f"Finished {c} states: {f:>{num}}/{n} = {p:4.1f} %, total: {t}, remaining: {e}"


def count_eigenvalues(matrix: sp.SparseMatrix, factor: int) -> list:
    """ Calculate and return a sorted list of all unique eigenvalues of the given symbolic SymPy matrix and their
    respective multiplicity. The given factor is used to determine identical eigenvalues. Each eigenvalue must become
    an integer when multiplied by this factor. """

    # Convert symbolic SymPy matrix into a floating point numpy array
    M = np.array(matrix.tolist()).astype(float)

    # Get numerical eigenvalues of the matrix. Note that we must use the algorithm for general non-symmetric
    # matrices and assert real eigenvalues manually, because our main algorithm is not orthonormalizing the
    # transformation matrices before the very end.
    eigenvalues = np.linalg.eigvals(M)
    is_real = np.max(np.abs(np.imag(eigenvalues))) < ATOL
    assert is_real
    eigenvalues = np.real(eigenvalues)

    # Identify identical eigenvalues by using the fact that all eigenvalues must be integers when multiplied by
    # the given factor.
    eigenvalues *= factor
    is_integer = np.all(np.isclose(eigenvalues, np.round(eigenvalues), atol=ATOL))
    assert is_integer
    eigenvalues = np.round(eigenvalues).astype(int)

    # Build and return a sorted list of all unique symbolic SymPy eigenvalues and their respective multiplicity
    unique_values = []
    for eigenvalue in sorted(list(set(eigenvalues))):
        count = int(np.sum(np.array(eigenvalues) == eigenvalue))
        unique_values.append((sp.S(eigenvalue) / factor, count))
    return unique_values


def matrix_diagonalize(matrix: sp.SparseMatrix, factor: int):
    """ Generator function for eigenvalues and eigenvectors of each eigenspace of the given full-rank MxM SymPy
    matrix. The given factor is used to convert numerically calculated eigenvalues into integers. For each
    eigenspace of size N the function yields the eigenvalue, a MxN matrix of eigenvectors and a NxM matrix of
    inverse eigenvectors. Note that the eigenvectors are neither orthogonal nor normalized. The total size of
    all eigenspaces is M. """

    # Shortcut for the special case of a 1 x 1 matrix
    if matrix.shape == (1, 1):
        eigenvalue = matrix[0, 0]
        eigenvectors = eigenvectors_inv = sp.ones(1, 1)
        yield eigenvalue, eigenvectors, eigenvectors_inv

    # Other matrix sizes
    else:

        # Determine all unique eigenvalues and their multiplicity
        unique_values = count_eigenvalues(matrix, factor)

        # Sort for decreasing multiplicity. The order influences the overall speed of the algorithm. Not optimized
        # yet, but this order works fine in a reasonable amount of time.
        unique_values = sorted(unique_values, key=lambda x: (-x[1], x[0]))

        # Determine and yield the eigenspace for each unique eigenvalue
        transform = None
        transform_inv = None
        finished = 0
        for i, (eigenvalue, count) in enumerate(unique_values):

            # Eigenvectors for the given eigenvalue (null space)
            if i == 0:
                M = matrix
            else:
                M = transform_inv[finished:, :] * matrix * transform[:, finished:]
            M = M - eigenvalue * sp.eye(M.shape[0])
            M_nul = M.nullspace()
            assert len(M_nul) == count

            # Add remaining base vectors (column space) to complete the transformation matrix
            if len(unique_values) == 1:
                V = sp.Matrix.hstack(*M_nul)
            else:
                M_col = M.columnspace()
                V = sp.Matrix.hstack(*M_nul, *M_col)

            # Back transform to product space and inversion of the transformation matrix
            if finished == 0:
                transform = V
                transform_inv = V.inv()
            else:
                transform[:, finished:] = transform[:, finished:] * V
                transform_inv[finished:, :] = V.inv() * transform_inv[finished:, :]

            # Yield eigenvalue and eigenvectors of the current eigenspace
            eigenvectors = transform[:, finished:finished + count]
            eigenvectors_inv = transform_inv[finished:finished + count, :]
            finished += count
            yield eigenvalue, eigenvectors, eigenvectors_inv


def transform_level(result, level, label="", transform=None, transform_inv=None):
    """ Determine the eigenspaces of the symmetry operator matrix of this level and recursively determine the
    eigenspaces of the symmetry operator matrices of all higher levels in each eigenspace of this level. Store
    the eigenvalues and eigenvectors of each eigenspace of this level in the result object. """

    # Symmetry operator matrix of this diagonalization level
    matrix = result.matrix(level)

    # Apply the transformation obtained from all previous levels
    if level > 0:
        matrix = transform_inv * matrix * transform

    # The product of any eigenvalue of the symmetry operator matrix with this factor results in an integer value
    factor = result.factor(level)

    # Loop through every eigenspace of the symmetry operator matrix and get eigenvalue and eigenvectors of each space
    for eigenvalue, eigenvectors, eigenvectors_inv in matrix_diagonalize(matrix, factor):

        # Multiplicity of the eigenvalue of the current eigenspace
        count = eigenvectors.shape[1]

        # Store the eigenvalue with its multiplicity in the result object
        result.add_values(level, eigenvalue, count)

        # Log the space separated string of the irreducible representations from this and all previous levels
        this_label = f"{label} {SYM_INFO[result.name(level)]['repr'](eigenvalue)}"
        logger.debug(f"{result.status} |{this_label} >")

        # Recursion to next level
        if level < result.max_level:

            # Update the transformation matrix with the eigenvectors of the current eigenspace
            if level == 0:
                this_transform = eigenvectors
                this_transform_inv = eigenvectors_inv
            else:
                this_transform = transform * eigenvectors
                this_transform_inv = eigenvectors_inv * transform_inv

            # Determine the eigenspaces of the symmetry matrices of all higher levels in the current eigenspace
            transform_level(result, level + 1, this_label, this_transform, this_transform_inv)

        # Final level
        else:

            # Update the transformation matrix with the eigenvectors of the current eigenspace
            assert level > 0
            vectors = transform * eigenvectors

            # Orthogonalize the eigenvectors
            vectors = [vectors.col(i) for i in range(vectors.shape[1])]
            if len(vectors) > 1:
                vectors = sp.GramSchmidt(vectors, orthonormal=False)

            # Normalize the eigenvectors
            vectors = [vector.normalized() for vector in vectors]

            # Store the eigenvectors in the result object
            result.add_vectors(vectors)


def transform_states(config):
    """ Calculate and return the sparse SymPy transformation matrix from electron product states to LS states and a
    list of eigenvalue dictionaries of all states. """

    # Initialise the results collector object
    result = Result(config)

    # Determine all eigenspaces (symbolic diagonalisation algorithm)
    transform_level(result, 0)
    transform = result.transform
    t = result.total_time
    logger.info(f"Prepared {config.name} transformation matrix ({t:.1f} seconds)")

    # Build a dictionary containing lists of the eigenvalues of all states for each symmetry operator name
    eigenvalues = dict(zip(result.names, result.eigenvalues))

    # Sort states in canonical order
    transform, eigenvalues = sort_states(config, transform, eigenvalues)

    # Add certain classification numbers to the dictionary of eigenvalues
    eigenvalues = classify_states(config, eigenvalues)

    # Build a list which contains a dictionary of all eigenvalues and classification numbers for each state
    # in LS coupling
    states = get_states(config, eigenvalues)

    # Fix the global signs of the states in each J eigenspace
    transform = correct_signs(config, transform, eigenvalues)

    # Send the list of all LS terms to the log
    terms = str_terms(config, states)
    for i, term in enumerate(terms):
        key = f"{config.name} term {i}:"
        logging.debug(f"  {key:<12} | {term} >")

    # Return the transformation matrix and the list of state dictionaries
    logger.info(f"Finished {config.name} transformation matrix")
    return sp.SparseMatrix(transform), states


##########################################################################
# LS states class
##########################################################################

class LS_States:
    """ This class represents a list of electron states in LS coupling. """

    def __init__(self, config: Config, eigenvalues: dict, indices: list):

        # State space string
        self.state_space = "SLJM"

        # Chain of tensor operator keys used for the state classification
        self.tensor_chain = SYM_CHAIN[config_key(config)]["chain"]
        self.tensor_description = {name: SYM_INFO[name]["desc"] for name in self.tensor_chain}

        # Dictionaries of eigenvalues and their irreducible representations
        self.eigenvalues = eigenvalues
        self.representations = {t: [SYM_INFO[t]["repr"](v) for v in values] for t, values in eigenvalues.items()}

        # Correlation space of state signs (restricts coupling)
        self.global_signs = "J space"

        # List of states as sequences of eigenvalue indices along the chain of tensor operators
        self.indices = indices
        self.num_states = len(indices)

        # List of full set of irreducible representations for all states
        states_dict = get_states(config, self.eigenvalue_lists())
        self.names = str_terms(config, states_dict, "template_full")

    @classmethod
    def from_meta(cls, states_dict, info_meta):
        """ Return a LS_States object initialized from its data container dictionaries. """

        # Initialize empty LS_States object
        states = cls.__new__(cls)

        # Extract state info
        states.state_space = info_meta["stateSpace"]
        states.tensor_chain = info_meta["tensorChain"]
        states.tensor_description = info_meta["tensorDescription"]
        states.eigenvalues = {t: [sp.S(value) for value in v] for t, v in info_meta["eigenvalues"].items()}
        states.representations = info_meta["irreducibleRepresentations"]
        states.names = info_meta["stateNames"]
        states.global_signs = info_meta["globalSigns"]
        states.num_states = info_meta["numStates"]

        # Extract list of states
        states.indices = decode_uint_array(states_dict, "indices")

        # Sanity checks for redundant information
        assert states.state_space in ("SLJM", "SLJ")
        assert states.state_space == states_dict["stateSpace"]
        assert states.num_states == len(states.indices)

        # Return LS_States object
        return states

    def as_meta(self):
        """ Return the data container dictionaries representing this object. """

        # States dictionary
        states_dict = encode_uint_array(self.indices, "indices")
        states_dict["tensorChain"] = self.tensor_chain
        states_dict["stateSpace"] = self.state_space

        # States info dictionary
        info_meta = {
            "stateSpace": self.state_space,
            "tensorChain": self.tensor_chain,
            "tensorDescription": self.tensor_description,
            "eigenvalues": {t: list(map(str, values)) for t, values in self.eigenvalues.items()},
            "irreducibleRepresentations": self.representations,
            "stateNames": self.names,
            "globalSigns": self.global_signs,
            "numStates": self.num_states,
        }

        # Return dictionaries
        return states_dict, info_meta

    @staticmethod
    def hash_data(hasher, states_dict, info_meta):
        """ Update hasher with representative state data. """

        # Update hasher with states_dict
        for key in sorted(states_dict.keys()):
            if key in ["tensorChain"]:
                continue
            value = states_dict[key]
            hasher.update(key.encode('utf-8'))
            if key == "stateSpace":
                hasher.update(value.encode('utf-8'))
            else:
                hasher.update(value.tobytes())

        # Update hasher with info_meta
        for key in sorted(info_meta.keys()):
            if key in ["tensorChain", "tensorDescription", "irreducibleRepresentations", "stateNames"]:
                continue
            value = info_meta[key]
            hasher.update(key.encode('utf-8'))
            if key == "eigenvalues":
                for tensor in sorted(value.keys()):
                    hasher.update(tensor.encode('utf-8'))
                    hasher.update(str(value[tensor]).encode('utf-8'))
            else:
                hasher.update(str(value).encode('utf-8'))

    def state_eigenvalues(self, names=None) -> list:
        """ Return a list of eigenvalue dictionaries of all states. Keys of the dictionaries are either the given
        tensor names or the full chain of symmetry tensors. """

        # Tensor names of the eigenvalue dictionaries
        if names is None:
            names = self.tensor_chain
        else:
            assert all(name in self.tensor_chain for name in names)

        # Prepare and return the eigenvalue dictionaries
        eigenvalues = self.eigenvalue_lists()
        states = [{name: eigenvalues[name][i] for name in names} for i in range(self.num_states)]
        return states

    def eigenvalue_lists(self, names=None) -> dict:
        """ Return a dictionary with tensor names as keys and the respective lists of eigenvalues of all states
        as values. Keys of the dictionaries are either the given tensor names or the full chain of symmetry
        tensors. """

        # Tensor names of the eigenvalue dictionary
        if names is None:
            names = self.tensor_chain
        else:
            assert all(name in self.tensor_chain for name in names)

        # Prepare and return the eigenvalue dictionary
        values = {}
        for name in names:
            i = self.tensor_chain.index(name)
            values[name] = [self.eigenvalues[name][state[i]] for state in self.indices]
        return values

    def state_representations(self, names=None) -> list:
        """ Return a list of irreducible representations dictionaries of all states. Keys of the dictionaries are
        either the given tensor names or the full chain of symmetry tensors."""

        # Tensor names of the representation dictionaries
        if names is None:
            names = self.tensor_chain
        else:
            assert all(name in self.tensor_chain for name in names)

        # Prepare and return the representation dictionaries
        representations = self.representation_lists()
        states = [{name: representations[name][i] for name in names} for i in range(self.num_states)]
        return states

    def representation_lists(self, names=None) -> dict:
        """ Return a dictionary with tensor names as keys and the respective lists of irreducible representations
        of all states as values. Keys of the dictionaries are either the given tensor names or the full chain of
        symmetry tensors. """

        # Tensor names of the representation dictionary
        if names is None:
            names = self.tensor_chain
        else:
            assert all(name in self.tensor_chain for name in names)

        # Prepare and return the representation dictionary
        values = {}
        for name in names:
            i = self.tensor_chain.index(name)
            values[name] = [self.representations[name][state[i]] for state in self.indices]
        return values

    def state_spaces(self, name):
        """ Return all eigenspaces of the given symmetry tensor as (start, stop) index tuples. """

        assert name in self.tensor_chain
        names = self.tensor_chain[:self.tensor_chain.index(name) + 1]
        states = self.state_eigenvalues(names)
        spaces = []
        i = 0
        for j in range(1, self.num_states + 1):
            if j == self.num_states or states[j] != states[i]:
                spaces.append((i, j))
                i = j
        return spaces

    def indices_j(self):
        """ Return the indices of all stretched states with M = J. """

        assert self.state_space == "SLJM"
        return [j - 1 for i, j in self.state_spaces("J2")]

    def collapse_j(self):
        """ Return this states object with collapsed J spaces. """

        # Sanity checks
        assert self.state_space == "SLJM"
        assert self.tensor_chain[-1] == "Jz"

        # Indices of stretched states with M = J
        indices = self.indices_j()

        # Get dictionary representation of the current states object
        states_dict, info_meta = self.as_meta()

        # Remove the Jz part from the eigenvalue indices
        assert "indices" in states_dict
        states_dict["indices"] = states_dict["indices"][indices, :-1]
        states_dict["stateSpace"] = "SLJ"

        # Remove Jz from the states metadata
        info_meta["stateSpace"] = "SLJ"
        info_meta["tensorChain"] = info_meta["tensorChain"][:-1]
        del info_meta["tensorDescription"]["Jz"]
        del info_meta["eigenvalues"]["Jz"]
        del info_meta["irreducibleRepresentations"]["Jz"]
        info_meta["stateNames"] = [info_meta["stateNames"][i].rpartition(" ")[0] for i in indices]
        info_meta["globalSigns"] = "None"
        info_meta["numStates"] = len(indices)

        # Build and return a new states object
        return self.from_meta(states_dict, info_meta)


###########################################################################
# Transform class
###########################################################################

TITLE = "Transformation matrix from product states to LS coupling"

DESCRIPTION = """
This container contains the exact symbolic orthonormal transformation matrix V from electron product states to LS 
coupling for the electron configuration {config_name}.
An operator matrix M can be transformed from the product to the LS state space by calculating the matrix product
M' = V^T * M * V, with the transposed transformation matrix V^T.
The signs of the LS states are correctly correlated within the J spaces, which allows to build superpositions of
Stark states with same J, but different M in intermediate coupling based on these LS states.
<br> {row_desc}
<br> {col_desc}
<br> {matrix_desc}
"""

# Description of the HDF5 container item holding the LS states of a configuration
SLJM_DESC = """
The HDF5 item '{states_hdf5}' contains all electron states in LS coupling.
Each state (row) within this array is a sequence of indices following the chain of tensor operator names in the
attribute 'tensorChain' and referencing the respective eigenvalue in '{states}.eigenvalues' in the JSON item {json}.
Eigenvalues are short rational numbers given as strings.
The attribute {states} of the JSON item '{json}' also contains the respective irreducible representations as well as a
description of the tensor operator for each of the names and string representations for all states.
"""

# Description of the HDF5 container item holding the LS states of a configuration with collapsed J spaces
SLJ_DESC = """
The HDF5 item '{states_hdf5}' contains all electron states in LS coupling with collapsed J spaces.
Each state (row) within this array is a sequence of indices following the chain of tensor operator names in the
attribute 'tensorChain' and referencing the respective eigenvalue in '{states}.eigenvalues' in the JSON item {json}.
Eigenvalues are short rational numbers given as strings.
The attribute {states} of the JSON item '{json}' also contains the respective irreducible representations as well as a
description of the tensor operator for each of the names and string representations for all states.
"""


class TransformContainer(Vault):
    """ Class representing a transform data container. """

    # Description of the HDF5 container item holding the LS states of a configuration required by the transformation
    # interface
    states_desc = {"SLJM": SLJM_DESC, "SLJ": SLJ_DESC}

    def __init__(self, config_name):
        """ Provide the data container. """

        # Configuration string
        self.config_name = config_name

        # Load or generate data container
        self.file = self.get_path(self.config_name)
        self.update_container(self.file, __version__)

    def generate_container(self, dc=None):
        """ Generate the LS transformation matrix and store it in a data container file. """

        if dc:
            v = dc["meta.json"]["version"]
            logger.info(f"Update {self.config_name} transformation matrix (version {v} -> {__version__})")
        else:
            logger.info(f"Generate {self.config_name} transformation matrix (version {__version__})")
        t_all = time.time()

        # Initialize the data hasher
        hasher = hashlib.sha256()

        # Get electron configuration
        config = Config(self.config_name)
        config_meta = config.info.as_meta()

        # Row states dictionary and metadata (Product)
        row_states, row_meta = config.states.as_meta()
        config.states.hash_data(hasher, row_states, row_meta)

        # Get LS transformation matrix, eigenvalues, and state indices from data container or determine from scratch
        if dc:
            state_matrix = SymMatrix.from_meta(dc["data/matrix.hdf5"], dc["data/transform.json"]["matrix"])
            indices = decode_uint_array(dc["data/col_states.hdf5"], "indices")
            eigenvalues = dc["data/transform.json"]["col_states"]["eigenvalues"]
            eigenvalues = {t: [sp.S(value) for value in values] for t, values in eigenvalues.items()}
        else:
            # Build transformation matrix and store it as SparseMatrix object
            transform, states_dict = transform_states(config)
            logger.debug("Creating StateMatrix.")
            t = time.time()
            state_matrix = SymMatrix.from_matrix("Product", "SLJM", transform)
            t = time.time() - t
            logger.debug(f"Built StateMatrix. ({t:.1f} seconds)")

            # Prepare eigenvalues and state indices
            chain = SYM_CHAIN[config_key(config)]["chain"]
            eigenvalues = {t: sorted(list(set(state[t] for state in states_dict))) for t in chain}
            indices = [[eigenvalues[t].index(state[t]) for t in chain] for state in states_dict]

        # Row states dictionary and metadata (LS)
        states = LS_States(config, eigenvalues, indices)
        col_states, col_meta = states.as_meta()
        states.hash_data(hasher, col_states, col_meta)

        # Dictionary and metadata of transformation matrix
        matrix_dict, matrix_meta = state_matrix.as_meta()
        SymMatrix.hash_data(hasher, matrix_dict)

        # Generate data hash
        data_hash = hasher.hexdigest()
        if dc and "sha256Data" in dc["content.json"] and data_hash != dc["content.json"]["sha256Data"]:
            raise VersionError

        # Prepare container description string
        kwargs = {
            "config_name": self.config_name,
            "matrix_hdf5": "matrix.hdf5",
            "matrix": "orthonormal transformation matrix",
            "row_hdf5": "row_states.hdf5",
            "col_hdf5": "col_states.hdf5",
            "json": "transform.json",
        }
        row_kwargs = {"states": "row_states", "states_hdf5": "row_states.hdf5"}
        kwargs["row_desc"] = desc_format(config.states_desc, kwargs | row_kwargs)
        col_kwargs = {"states": "col_states", "states_hdf5": "col_states.hdf5"}
        kwargs["col_desc"] = desc_format(self.states_desc["SLJM"], kwargs | col_kwargs)
        kwargs["matrix_desc"] = desc_format(SymMatrix.meta_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Initialise container structure
        items = {
            "content.json": {
                "containerType": {"name": "ameliTransform"},
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
            "data/transform.json": {
                "version": __version__,
                "dataType": "symbolic",
                "config": config_meta,
                "row_states": row_meta,
                "col_states": col_meta,
                "matrix": matrix_meta,
            },
            "data/row_states.hdf5": row_states,
            "data/col_states.hdf5": col_states,
            "data/matrix.hdf5": matrix_dict,
        }

        # Create data container and store in file
        self.write_container(self.file, items)
        t = time.time() - t_all
        logger.info(f"Stored {self.config_name} transformation matrix ({t:.1f} seconds) -> {self.file}")

    @staticmethod
    def get_path(config_name):
        """ Return data container file name. """

        return f"{config_name}/transform.zdc"


class Transform(TransformContainer):
    """ Class of the exact symbolic transformation matrix from product states to LS coupling. It provides the SymPy
    matrix in the attribute 'matrix' and the respective floating point NumPy array from the method 'array(dtype)'. """

    # Transformation matrix
    matrix: sp.Matrix = None

    def __init__(self, config_name):
        """ Initialize the orthonormal transformation matrix from the product space to LS coupling. """

        # Configuration string
        self.config_name = config_name

        # Load or generate data container
        super().__init__(self.config_name)
        dc = self.read_container(self.file)

        # Extract UUID and code version from the container
        meta = dc["data/transform.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Sanity check for data type
        assert meta["dataType"] == "symbolic"

        # Extract electron configuration from the container
        self.config = ConfigInfo.from_meta(meta["config"])
        assert self.config.name == config_name

        # Extract row states (Product) and column states (SLJM)
        self.row_states = space_registry["Product"].from_meta(dc["data/row_states.hdf5"], meta["row_states"])
        self.col_states = self.states_from_meta(dc["data/col_states.hdf5"], meta["col_states"])
        assert self.row_states.num_states == self.col_states.num_states
        self.num_states = self.row_states.num_states

        # Extract transformation matrix
        self.info = SymMatrix.from_meta(dc["data/matrix.hdf5"], meta["matrix"])
        self.matrix = self.info.matrix
        assert self.info.row_space == self.row_states.state_space
        assert self.info.col_space == self.col_states.state_space

    @staticmethod
    def states_from_meta(states_dict, info_meta):
        """ Return a LS_States object initialized from its data container dictionaries. """

        return LS_States.from_meta(states_dict, info_meta)

    def states_as_meta(self):
        """ Return the data container dictionaries representing the states in LS coupling. """

        return self.col_states.as_meta()

    @staticmethod
    def hash_data(hasher, states_dict, info_meta):
        """ Update hasher with representative state data. """

        LS_States.hash_data(hasher, states_dict, info_meta)


# Register space of electron states in LS coupling
register_space("SLJM", Transform)
register_subspace("SLJM", "SLJ")
