##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Transform, which represents the
# orthonormal transformation matrix V from electron product states to
# LS coupling for a given many-electron configuration. The signs of the
# LS states are correctly correlated within the J spaces, which allows
# to build superpositions of Stark states with same J, but different M
# in intermediate coupling based on these LS states.
#
# The module also provides the class SljmStates, which represents all
# LS states of a configuration.
#
##########################################################################

import logging
import time
from functools import lru_cache

import numpy as np
import sympy as sp

from . import space_registry, register_space, desc_format
from .uintarray import decode_uint_array, encode_uint_array
from .datatype import DataType
from .casimir import CASIMIR
from .vault import get_vault
from .config import SPECTRAL, ConfigInfo, get_config
from .unit import Unit
from .matrix import Matrix

__version__ = "1.0.0"
ATOL = 1e-12

###########################################################################
# LS term classification
###########################################################################

SYM_INFO = {
    "S2": {"matrix": "S2", "repr": lambda x: str(sp.sqrt(4 * x + 1)), "factor": 4,
           "desc": "Squared operator of the total spin, represented by the multiplicity S(S+1)"},
    "C7": {"matrix": "C7", "repr": lambda x: CASIMIR["SO7"].key(x), "factor": 5,
           "desc": "Casimir operator of the special orthogonal group in 7 dimensions SO(7)"},
    "C5": {"matrix": "C5", "repr": lambda x: CASIMIR["SO5"].key(x), "factor": 3,
           "desc": "Casimir operator of the special orthogonal group in 5 dimensions SO(5)"},
    "C2": {"matrix": "C2", "repr": lambda x: CASIMIR["G2"].key(x), "factor": 12,
           "desc": "Casimir operator of the special group G_2"},
    "L2": {"matrix": "L2", "repr": lambda x: SPECTRAL[(sp.sqrt(4 * x + 1) - 1) / 2].upper(), "factor": 4,
           "desc": "Squared operator of the total orbital angular momentum, represented by its spectral character"},
    "J2": {"matrix": "J2", "repr": lambda x: str((sp.sqrt(4 * x + 1) - 1) / 2), "factor": 4,
           "desc": "Squared operator of the total angular momentum, represented by the quantum number J"},
    "Jz": {"matrix": "Jz", "repr": lambda x: ("+", "")[int(bool(x < 0))] + str(x), "factor": 2,
           "desc": "Operator of the z component of the total angular momentum, represented by the quantum number M_J"},
    "sen": {"matrix": None, "repr": lambda x: str(x), "factor": None,
            "desc": "Seniority number"},
    "tau": {"matrix": None, "repr": lambda x: "" if x == 0 else chr(ord("A") + x - 1), "factor": None,
            "desc": "Index of different states with same set of quantum numbers"},
    "num": {"matrix": None, "repr": lambda x: "" if x == 0 else str(x), "factor": None,
            "desc": "Index of different LS states with same L and S quantum numbers"},
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

def reduced(dtype, index_a, index_b, operator, k, transform, J, Ma, Mb):
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
    denominator = dtype.sym3j(J, k, J, -Ma, q, Mb)
    if denominator == 0:
        return sp.nan
    if (J - Ma) % 2:
        denominator = -denominator

    # Matrix element of the q-component of the tensor operator in LS coupling
    if dtype.is_symbolic:
        numerator = transform.col(index_a).T * operator[q] * transform.col(index_b)
        assert numerator.shape == (1, 1)
        numerator = numerator[0, 0]
    else:
        numerator = transform[:, index_a].T @ operator[q] @ transform[:, index_b]
    if dtype.is_zero(numerator, 4):
        return sp.nan

    # Return valid non-zero reduced matrix element
    return numerator / denominator


def update_signs(dtype, slices, operator, k, transform, J, M, known, signs):
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

        # Take the reduced matrix of the stretched state M = -J as reference, if it is not zero
        diagonal = reduced(dtype, i, i, operator, k, transform, J[i], M[i], M[i])
        if diagonal is sp.nan:
            continue
        known[i] = True

        # Loop through all other eigenvectors
        for col in range(i + 1, j):

            # Sign was already fixed
            if known[col]:
                continue

            # Loop through all rows < col (negative coordinates of the tensor operator) limited by the tensor rank
            for row in range(max(i, col - k), col)[::-1]:

                # Try to get a value for the reduced matrix element
                assert J[row] == J[col]
                element = reduced(dtype, row, col, operator, k, transform, J[row], M[row], M[col])
                if element is sp.nan:
                    continue

                # Apply sign of the row eigenvector
                if signs[row]:
                    element = -element

                # Fix sign of this eigenvector and proceed with the next one
                if dtype.is_zero(element - diagonal, 100):
                    signs[col] = 0
                elif dtype.is_zero(element + diagonal, 100):
                    signs[col] = 1
                else:
                    print(element, diagonal)
                    raise RuntimeError("Wigner-Eckart theorem failed!")
                known[col] = True
                break


def correct_signs(dtype, config, transform: sp.Matrix, eigenvalues: dict) -> sp.Matrix:
    """ Use the Wigner-Eckart theorem in the JM space to obtain consistent global signs of the eigenvectors in each
    J eigenspace. This allows to construct correct superpositions of M states. The matrix transform is returned with
    adjusted signs for all column vectors. """

    logger = logging.getLogger()
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
    names = SYM_CHAIN[config_key(config)]["chain"][:-1]
    keys = [state_key(eigenvalues, i, names) for i in range(num_states)]
    slices = []
    i = 0
    for j in range(num_states + 1):
        if j == num_states or keys[j] != keys[i]:
            slices.append((i, j))
            i = j

    # Use unit tensor operators to fix the global signs
    l = max(electron.l for electron in config.states.electron_pool)
    for k in range(1, 2 * l + 1):

        # Use the components of the unit tensor operator of rank k in the orbital angular momentum space to correct
        # as many signs as possible. The function update_signs() uses only non-positive components.
        operator = {q: Unit(dtype, config.name, "UT/{k},0,{k},{q}".format(k=k, q=q)).matrix for q in range(-k, 1)}
        update_signs(dtype, slices, operator, k, transform, J, M, known, signs)
        logger.info(f"Global signs U({k}) in {config.name}: {sum(known)}/{len(known)} fixed")
        if np.all(known):
            break

        # Operators in the spin space are limited to rank 1
        if k > 1:
            continue

        # Use the unit tensor operator of rank k in the spin angular momentum space to correct as many signs
        # as possible
        operator = {q: Unit(dtype, config.name, "UT/0,{k},{k},{q}".format(k=k, q=q)).matrix for q in range(-k, 1)}
        update_signs(dtype, slices, operator, k, transform, J, M, known, signs)
        logger.info(f"Global signs T({k}) in {config.name}: {sum(known)}/{len(known)} fixed")
        if np.all(known):
            break

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


def get_sym_matrices(dtype, config):
    """ Return three lists: the tensor keys of the symmetry chain, all symmetry matrices and all factors to obtain
    integer eigenvalues. """

    # Load or generate the symmetry operator matrix for every diagonalizing level
    chain = SYM_CHAIN[config_key(config)]["chain"]
    names = [name for name in chain if SYM_INFO[name]["matrix"]]
    matrices = [Matrix(dtype, config.name, SYM_INFO[name]["matrix"], "Product").matrix for name in names]

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

    def __init__(self, dtype, config):

        # Store data type
        self.dtype = dtype

        # Store the Config object which is required for the calculation of the symmetry matrices
        self.config = config

        # Number of electron states of the given configuration
        self.num_states = self.config.num_states

        # Tensor keys of the symmetry chain, symmetry matrices and factors to obtain integer eigenvalues
        self.names, self.matrices, self.factors = get_sym_matrices(self.dtype, self.config)

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

    def matrix(self, level: int) -> sp.Matrix:
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


def count_eigenvalues(matrix: sp.Matrix, factor: int) -> list:
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


def matrix_diagonalize(matrix: sp.Matrix, factor: int):
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


def transform_level(logger, result, level, label="", transform=None, transform_inv=None):
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
            transform_level(logger, result, level + 1, this_label, this_transform, this_transform_inv)

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


def transform_sym(logger, dtype, config):
    result = Result(dtype, config)

    transform_level(logger, result, 0)
    # result.sort()

    # Transformation matrix with eigenvectors as columns
    transform = result.transform

    # Build a dictionary containing lists of the eigenvalues of all states for each symmetry operator name
    eigenvalues = dict(zip(result.names, result.eigenvalues))

    t = result.total_time
    logger.info(f"Prepared {config.name} transformation matrix ({t:.1f} seconds)")

    return transform, eigenvalues


def transform_num(logger, dtype, config):
    t = time.time()

    names, matrices, factors = get_sym_matrices(dtype, config)

    # Initialize the eigenvalue matrix, the transformation matrix, and the dictionary of SymmetryList objects
    transform = None
    eigenvalues = {}

    # Initialize the list of sub-spaces which will be split by the algorithm
    slices = [(0, config.num_states)]

    # Follow the chain or symmetry operators, but skip the pseudo operators "tau" and "num". Build a transformation
    # matrix from product states to SLJM coupling together with the eigenvalues of all symmetry operators.
    for name, matrix, factor in zip(names, matrices, factors):

        # Get the matrix of the current symmetry operator in the determinantal product space and apply the current
        # transformation matrix
        if transform is None:
            M = matrix
        else:
            M = transform.T @ matrix @ transform

        # Initialize eigenvalues and eigenvectors of the current symmetry operator
        values = []
        eigenvectors = dtype.eye(config.num_states)

        # Calculate eigenvalues and eigenvectors of the current symmetry operator by diagonalising its pre-transformed
        # matrix in each of the current sub-spaces
        for i, j in slices:
            if j - i > 1:
                V, U = np.linalg.eigh(M[i:j, i:j])
                values.extend(list(V))
                eigenvectors[i:j, i:j] = U
            elif j - i == 1:
                values.append(M[i, i])
            else:
                raise RuntimeError("Empty slice!")

        # Store SymmetryList object containing the eigenvalues of the current symmetry operator for all states
        values_float = dtype.array(values) * factor
        values = np.round(values_float).astype(int)
        assert np.allclose(values_float, values, atol=1000 * dtype.eps)
        values = [sp.S(value) / factor for value in values]

        # Split the sub-spaces in such a way that all states inside the new sub-spaces have the same eigenvalue of
        # the current symmetry operator
        slices_new = []
        for start, stop in slices:
            i = start
            for j in range(i + 1, stop + 1):
                if j == stop or values[j] != values[i]:
                    slices_new.append((i, j))
                    i = j
        slices = slices_new

        # Use the eigenvectors of the current symmetry operator to update the transformation matrix
        if transform is None:
            transform = np.array(eigenvectors)
        else:
            transform = transform @ eigenvectors

        eigenvalues[name] = values
        reprs = ", ".join(SYM_INFO[name]["repr"](x) for x in set(values))
        logger.info(f"  {config.name} | level {name} finished: {reprs} ({len(slices)} eigenspaces)")

    # Return orthonormal transformation matrix and eigenvalues
    t = time.time() - t
    logger.info(f"Prepared {config.name} transformation matrix ({t:.1f} seconds)")
    return transform, eigenvalues


def transform_states(logger, dtype, config):
    """ Calculate and return the SymPy transformation matrix from electron product states to LS states and a list
    of eigenvalue dictionaries of all states. """

    # Diagonalize all eigenspaces simultaneously
    if dtype.is_symbolic:
        transform, eigenvalues = transform_sym(logger, dtype, config)
    else:
        transform, eigenvalues = transform_num(logger, dtype, config)

    transform, eigenvalues = sort_states(config, transform, eigenvalues)

    # Add certain classification numbers to the dictionary of eigenvalues
    eigenvalues = classify_states(config, eigenvalues)

    # Build a list which contains a dictionary of all eigenvalues and classification numbers for each state
    # in LS coupling
    states = get_states(config, eigenvalues)

    # Fix the global signs of the states in each J eigenspace
    transform = correct_signs(dtype, config, transform, eigenvalues)

    # Send the list of all LS terms to the log
    terms = str_terms(config, states)
    for i, term in enumerate(terms):
        key = f"{config.name} term {i}:"
        logging.debug(f"  {key:<12} | {term} >")

    # Return the transformation matrix and the list of state dictionaries
    logger.info(f"Finished {config.name} transformation matrix")
    return transform, states


##########################################################################
# LS states class
##########################################################################

class SljmStates:
    """ This class represents a list of electron states in LS coupling. """

    # State space string
    state_space: str

    # Chain of tensor operator keys used for the state classification
    tensor_chain: list
    tensor_description: dict

    # Dictionaries of eigenvalues and their irreducible representations
    eigenvalues: dict
    representations: dict

    # Correlation space of state signs (restricts coupling)
    global_signs: str

    # List of states as sequences of eigenvalue indices along the chain of tensor operators
    indices: list
    num_states: int

    # List of full set of irreducible representations for all states
    names: list

    @classmethod
    def from_meta(cls, states_dict, info_meta):
        """ Return a SljmStates object initialized from its data container dictionaries. """

        # Initialize empty SljmStates object
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
        assert states.state_space == "SLJM"
        assert states.state_space == states_dict["stateSpace"]
        assert states.num_states == len(states.indices)

        # Return SljmStates object
        return states

    def as_meta(self):
        """ Return the data container dictionaries representing this object. """

        # States dictionary
        states_dict = encode_uint_array(self.indices, "indices")
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

    def terms(self, template="term") -> list:
        """ Return the list of unique LS terms of this configuration. The template can be chosen to be either
        'short', 'term', or 'full'. """

        raise NotImplemented("Add configName to the state metadata to make this method available again!")
        # assert template in ("short", "term", "full")
        # config = get_config(self.config_name)
        # terms = str_terms(config, self.state_eigenvalues(), f"template_{template}")
        # return terms

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


###########################################################################
# Transform class
###########################################################################

TITLE = "Transformation matrix from product states to LS coupling"

DESCRIPTION = """
This container stores the orthonormal transformation matrix V from electron product states to LS coupling for a
given many-electron configuration.
An operator matrix M can be transformed from the product to the LS state space by calculating the matrix product
M' = V^T * M * V, with the transposed transformation matrix V^T.
The signs of the LS states are correctly correlated within the J spaces, which allows to build superpositions of
Stark states with same J, but different M in intermediate coupling based on these LS states.
<br> {row_desc}
<br> {col_desc}
<br> {matrix_desc}
"""

# Description of the HDF5 container item holding the LS states of a configuration
LS_DESC = """
The HDF5 item '{states_hdf5}' contains all electron states in LS coupling.
Each state (row) within this array is a sequence of indices following the chain of tensor operator names in the
attribute 'tensorChain' and referencing the respective eigenvalue in '{states}.eigenvalues' in the JSON item {json}.
Eigenvalues are short rational numbers given as strings.
The attribute {states} of the JSON item '{json}' also contains the respective irreducible representations as well as a
description of the tensor operator for each of the names and string representations for all states.
"""


class Transform:
    """ Class of the transformation matrix from product states to LS coupling. It provides the SymPy matrix in the
    attribute 'matrix' and the respective floating point NumPy array from the method 'array(dtype)'. """

    # Description of the HDF5 container item holding the LS states of a configuration required by the transformation
    # interface
    states_desc = LS_DESC

    # Transformation matrix
    matrix: sp.Matrix | np.ndarray = None

    def __init__(self, dtype, config_name):
        """ Initialize the orthonormal transformation matrix from the product space to LS coupling. """

        # Store data type
        if isinstance(dtype, str):
            dtype = DataType(dtype)
        self.dtype = dtype

        # Configuration string
        self.config_name = config_name

        # Data container cache and container file name
        self.vault = get_vault(self.config_name)
        self.file = f"{dtype.name}/transform.zdc"

        # Load or generate data container
        if self.file not in self.vault:
            self.generate_container()
        dc = self.vault[self.file]

        # Extract UUID and code version from the container
        meta = dc["data/transform.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Sanity check for data type
        assert self.dtype.name == meta["dataType"]

        # Extract electron configuration from the container
        self.config = ConfigInfo.from_meta(meta["config"])
        assert self.config.name == config_name

        # Extract row states (Product) and column states (SLJM)
        self.row_states = space_registry["Product"].from_meta(dc["data/row_states.hdf5"], meta["row_states"])
        self.col_states = self.states_from_meta(dc["data/col_states.hdf5"], meta["col_states"])
        assert self.row_states.num_states == self.col_states.num_states
        self.num_states = self.row_states.num_states

        # Extract transformation matrix
        self.info = self.dtype.from_meta(dc["data/matrix.hdf5"], meta["matrix"])
        self.matrix = self.info.matrix
        assert self.info.row_space == self.row_states.state_space
        assert self.info.col_space == self.col_states.state_space

    def generate_container(self):
        """ Generate the LS transformation matrix and store it in a data container file. """

        logger = logging.getLogger()
        logger.info(f"Generating {self.config_name} transformation matrix")
        t_all = time.time()

        # Get electron configuration
        config = get_config(self.config_name)
        config_meta = config.info.as_meta()

        # Build transformation matrix and store it as SparseMatrix object
        transform, states_dict = transform_states(logger, self.dtype, config)
        logger.debug("Creating StateMatrix.")
        t = time.time()
        state_matrix = self.dtype.from_matrix("Product", "SLJM", transform)
        t = time.time() - t
        logger.debug(f"Built StateMatrix. ({t:.1f} seconds)")
        t = time.time()
        matrix_dict, matrix_meta = state_matrix.as_meta()
        t = time.time() - t
        logger.debug(f"Finished StateMatrix. ({t:.1f} seconds)")

        # List of string representations of all states
        terms = str_terms(config, states_dict, "template_full")

        # Chain of tensor operators
        chain = SYM_CHAIN[config_key(config)]["chain"]
        tensor_desc = {name: SYM_INFO[name]["desc"] for name in chain}

        # Eigenvalues and their representations
        eigenvalues = {t: sorted(list(set(state[t] for state in states_dict))) for t in chain}
        eigenvalues_str = {t: list(map(str, values)) for t, values in eigenvalues.items()}
        representations = {t: [SYM_INFO[t]["repr"](v) for v in values] for t, values in eigenvalues.items()}

        # Row states dictionary and metadata (Product)
        row_states, row_meta = config.states.as_meta()

        # Column states dictionary (SLJM)
        states = [[eigenvalues[t].index(state[t]) for t in chain] for state in states_dict]
        col_states = encode_uint_array(states, "indices")
        col_states["tensorChain"] = chain
        col_states["stateSpace"] = "SLJM"

        # Column states metadata (SLJM)
        col_meta = {
            "stateSpace": col_states["stateSpace"],
            "tensorChain": chain,
            "tensorDescription": tensor_desc,
            "eigenvalues": eigenvalues_str,
            "irreducibleRepresentations": representations,
            "stateNames": terms,
            "globalSigns": "J space",
            "numStates": len(states),
        }

        # Prepare container description string
        kwargs = {
            "matrix_hdf5": "matrix.hdf5",
            "matrix": "orthonormal transformation matrix",
            "dtype": self.dtype.name,
            "row_hdf5": "row_states.hdf5",
            "col_hdf5": "col_states.hdf5",
            "json": "transform.json",
        }
        row_kwargs = {"states": "row_states", "states_hdf5": "row_states.hdf5"}
        kwargs["row_desc"] = desc_format(config.states_desc, kwargs | row_kwargs)
        col_kwargs = {"states": "col_states", "states_hdf5": "col_states.hdf5"}
        kwargs["col_desc"] = desc_format(self.states_desc, kwargs | col_kwargs)
        kwargs["matrix_desc"] = desc_format(self.dtype.matrix_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Initialise container structure
        items = {
            "content.json": {
                "containerType": {"name": "ameliTransform"},
                "usedSoftware": [{"name": "AMELI", "version": "1.0.0",
                                  "id": "https://github.com/reincas/ameli", "idType": "URL"}],
            },
            "meta.json": {
                "title": TITLE,
                "description": description,
                "license": "cc-by-sa-4.0",
                "version": __version__,
            },
            "data/transform.json": {
                "version": __version__,
                "dataType": self.dtype.name,
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
        self.vault[self.file] = items
        t = time.time() - t_all
        logger.info(f"Stored {self.config_name} transformation matrix ({t:.1f} seconds) -> {self.file}")

    @staticmethod
    def states_from_meta(states_dict, info_meta):
        """ Return a SljmStates object initialized from its data container dictionaries. """

        return SljmStates.from_meta(states_dict, info_meta)

    def states_as_meta(self):
        """ Return the data container dictionaries representing the states in LS coupling. """

        return self.col_states.as_meta()


@lru_cache(maxsize=None)
def get_transform(dtype, config_name):
    """ Return cached Transform object. """

    return Transform(dtype, config_name)


# Register space of electron states in LS coupling
register_space("SLJM", Transform, get_transform)
