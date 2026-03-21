##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides classes representing exact symbolic
# signed square roots of rational numbers as SymPy objects:
#
#    expr = (-1)^s * sqrt(n/d)
#
# with two integer values n and d and a sign flag s.
#
# The class RationalRadical represents a single expression. It may be
# initialised directly from a given SymPy expression or from a
# dictionary using the classmethod from_dict(). Its method to_dict()
# exports its content into such a dictionary.
#
# The class RationalRadicalList represents a list of expressions. It may
# be initialized directly from a list of SymPy expressions or from a
# dictionary using the class method from_dict(). Its method to_dict()
# exports its content into such a dictionary. The class uses uint
# encoding and decoding from the module uintarray to allow the storage
# of numerators and denominators with unlimited bit-length in numpy
# arrays.
#
# The module also provides the class SymMatrix which represents a exact
# sparse symbolic matrix acting in given electron state spaces.
#
##########################################################################

import operator
from functools import reduce
import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_3j

from .uintarray import encode_uint_array, encode_uint_arrays, decode_uint_array


##########################################################################
# Square root of a rational number
##########################################################################

class RationalRadical:
    """ Data object representing a SymPy expression which can be written as the square root of a rational number
    '(-1)^s * sqrt(n/d)' with a sign flag s and two integers n and d."""

    def __init__(self, expr: sp.Expr | str):
        """ Initialize a new object from the given SymPy expression. """

        # Expression must simplify to a number
        expr = sp.sympify(expr)
        assert expr.is_number

        # Split the expression into four integers
        self.sign, self.numerator, self.denominator = self.split_sqrt_fraction(expr)

    @classmethod
    def from_dict(cls, args: dict) -> "RationalRadical":
        """ Create a RationalRadical object from a dictionary. """

        value = cls.__new__(cls)
        value.sign = args["sign"]
        value.numerator = args["numerator"]
        value.denominator = args["denominator"]
        return value

    def __str__(self):
        """ Return a string representation of the object. """

        return str(self.expr)

    def __eq__(self, other):
        """ Determine the equality of two RationalRadical objects. """

        assert isinstance(other, RationalRadical)

        return (self.sign == other.sign and
                self.numerator == other.numerator and
                self.denominator == other.denominator)

    @property
    def expr(self) -> sp.Expr:
        """ Return the object as SymPy expression 'expr = (-1)^s * sqrt(c/d)'. """

        expr = sp.sqrt(sp.Rational(self.numerator, self.denominator))
        if self.sign:
            return -expr
        return expr

    @classmethod
    def split_sqrt_fraction(cls, expr: sp.Expr) -> tuple:
        """ Split the given SymPy expression into three integers s, n and d with '(-1)^s * sqrt(c/d)'. Return the
        integers if possible and raise an AssertionError otherwise."""

        # Expression must simplify to a number, no symbols allowed
        expr = sp.sympify(expr)
        assert expr.is_number

        # Expression is '0'
        if expr.is_zero:
            return 0, 0, 1

        # Store the sign of the expression
        sign = int(expr.is_negative)

        # Numerator and denominator of the squared expression
        expr = expr * expr
        assert expr.is_rational
        numerator, denominator = expr.as_numer_denom()

        # Return the integer arguments
        return sign, numerator, denominator

    def as_dict(self) -> dict:
        """ Return a dictionary representation of the object. """

        return {
            "sign": self.sign,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }


##########################################################################
# List of square roots of rationals
##########################################################################

class RationalRadicalList:
    """ Data object representing a list of RationalRadical objects. It uses uint encoding and decoding from
    the module uintarray to allow the storage of numerators and denominators with unlimited bit-length in
    numpy arrays. """

    def __init__(self, expressions: list):
        """ Store given sequence of SymPy expressions as list of RationalRadical objects. """

        self.values = [RationalRadical(expr) for expr in expressions]

    @classmethod
    def from_dict(cls, meta: dict) -> "RationalRadicalList":
        """ Create a RationalRadicalList object from a uint-encoded dictionary representation. """

        obj = cls.__new__(cls)
        values = {
            "sign": decode_uint_array(meta, "sign"),
            "numerator": decode_uint_array(meta, "numerator"),
            "denominator": decode_uint_array(meta, "denominator"),
        }
        size = len(values["numerator"])
        values = [{key: int(values[key][i]) for key in values} for i in range(size)]
        obj.values = [RationalRadical.from_dict(value) for value in values]
        return obj

    def as_dict(self) -> dict:
        """ Return a uint-encoded dictionary representation of the object. """

        values = [value.as_dict() for value in self.values]
        values = {key: [value[key] for value in values] for key in values[0]}
        return encode_uint_arrays(values)


###########################################################################
# Symbolic state matrix class
###########################################################################

def is_sorted_lex(rows, columns):
    """ Return True if rows are non-decreasing and columns are increasing within each row (lexicographic order). """

    # Return False if row indices contain a backstep
    row_diff = np.diff(rows)
    if np.any(row_diff < 0):
        return False

    # Mask for all row indices identical to the next one
    same_row = row_diff == 0

    # Return False if column indices contain a backstep or stay constant where row indices are the same
    col_diff = np.diff(columns)
    if np.any(col_diff[same_row] <= 0):
        return False

    # Row and column indices are in lexicographic order
    return True


SYM_DESC = """
The HDF5 item '{matrix_hdf5}' contains the {matrix} elements as exact symbolic values.
The matrix is stored in a compressed format, which takes advantage of the sparsity of the matrix and its small number
of unique values.
The vector dataset 'rows' contains state indices related to the list of states in the item '{row_hdf5}'.
The vector dataset 'columns' contains state indices related to the list of states in the item '{col_hdf5}'.
For each row and column there is a value index in the dataset 'elements' which refers to the unique values
contained in the vector datasets 'signs', 'numerators', and 'denominators'.
The value of each matrix element is related to its sign flag s, numerator n, and denominator d by
value = (-1)^s * sqrt(n/d).
If the numerator or the denominator contain very large integers, the respective vector is split bit-wise in parts
named 'numerator_part<i>' or 'denominator_part<i>'.
The part with i=0 contains the lowest order bits of the values.
The size of the quadratic matrix is given by the square of the attribute 'numStates'.
The attribute 'isSymmetric' is True for a symmetric matrix. 
For a symmetric matrix the datasets contain only the elements on the diagonal and in the lower left triangle.
The attribute 'numElements' contains the number of non-zero matrix elements.
The attribute 'numDiagonalElements' contains the number of non-zero matrix elements on the matrix diagonal.
The attribute 'numUniqueValues' contains the number of different non-zero matrix element values.
"""


class SymMatrix:
    """ This class represents an exact symbolic sparse quadratic matrix. The list 'values' contains all unique
    non-zero element values. For each non-zero matrix element there is an element in the lists rows, columns,
    and elements. Each element in the list 'elements' is an index to the respective matrix element value in the
    list 'values'. """

    # State space strings
    row_space: str
    col_space: str

    # Number of electrons (matrix size is num_states * num_states)
    num_states: int

    # Empty matrix flag
    is_empty: bool

    # Symmetric matrix flag
    is_symmetric: bool

    # Number of non-zero matrix elements in total
    num_elements: int

    # Number of non-zero matrix elements on the matrix diagonal
    num_diagonal: int

    # Number of unique non-zero matrix elements
    num_unique: int

    # Matrix object
    matrix: sp.SparseMatrix

    # Description of meta data
    meta_desc = SYM_DESC

    def __init__(self, row_space: str, col_space: str, is_symmetric: bool, num_states: int):
        """ Initialize empty mutable SymMatrix object. """

        # Store number of electron states (matrix dimension)
        self.num_states = int(num_states)

        # Store names of row and column state spaces
        self.row_space = row_space
        self.col_space = col_space

        # Store symmetry flag
        self.is_symmetric = bool(is_symmetric)

        # Empty matrix
        self.matrix = sp.SparseMatrix(self.num_states, self.num_states, {})
        self.is_immutable = False

    def __setitem__(self, index: tuple, value):
        """ Store matrix element in the matrix object. """

        # Matrix must not be immutable
        assert not self.is_immutable

        # Store value in the matrix
        row, column = index
        self.matrix[row, column] = value
        if self.is_symmetric and row != column:
            self.matrix[column, row] = value

    def make_immutable(self):
        """ Determine and store matrix info and make the matrix immutable. """

        # Store or check info flags
        assert self.matrix.shape[0] == self.matrix.shape[1] == self.num_states
        self.is_empty = self.matrix.nnz() == 0
        if self.is_symmetric:
            assert self.is_symmetric == self.matrix.is_symmetric()

        # Statistics of non-zero elements
        self.num_elements = self.matrix.nnz()
        self.num_diagonal = self.matrix.diagonal().nnz()
        self.num_unique = len(set(self.matrix.todok().values()))

        # Matrix is immutable now
        self.is_immutable = True

    @classmethod
    def from_matrix(cls, row_space, col_space, matrix):
        """ Return an immutable SymMatrix object from a matrix object. """

        # Initialize empty matrix object
        num_states = matrix.shape[0]
        is_symmetric = matrix.is_symmetric()
        obj = cls(row_space, col_space, is_symmetric, num_states)

        # Store sparse SymPy matrix
        assert isinstance(matrix, sp.SparseMatrix)
        obj.matrix = matrix

        # Determine matrix info and make the matrix object immutable
        obj.make_immutable()

        # Return the matrix object
        return obj

    @classmethod
    def from_meta(cls, matrix_dict: dict, info_meta: dict):
        """ Return an immutable SymMatrix object from the data container dictionaries. """

        # Sanity check for redundant information
        for key, value in info_meta.items():
            assert matrix_dict[key] == info_meta[key]

        # Initialize empty matrix object
        row_space = info_meta["rowSpace"]
        col_space = info_meta["colSpace"]
        num_states = info_meta["numStates"]
        is_symmetric = info_meta["isSymmetric"]
        obj = cls(row_space, col_space, is_symmetric, num_states)

        # Extract and store matrix elements
        if not info_meta["isEmpty"]:
            rows = decode_uint_array(matrix_dict, "rows")
            columns = decode_uint_array(matrix_dict, "columns")
            elements = decode_uint_array(matrix_dict, "elements")
            values = [value.expr for value in RationalRadicalList.from_dict(matrix_dict).values]
            for row, col, element in zip(rows, columns, elements):
                obj[row, col] = values[element]

        # Determine matrix info and make the matrix object immutable
        obj.make_immutable()

        # Sanity checks
        assert obj.is_empty == info_meta["isEmpty"]
        assert obj.num_elements == info_meta["numElements"]
        assert obj.num_diagonal == info_meta["numDiagonalElements"]
        assert obj.num_unique == info_meta["numUniqueValues"]

        # Return matrix object
        return obj

    def as_meta(self):
        """ Return object content as data container dictionaries. """

        # Determine matrix info and make the matrix immutable
        if not self.is_immutable:
            self.make_immutable()

        # Matrix info dictionary
        info_meta = {
            "rowSpace": self.row_space,
            "colSpace": self.col_space,
            "numStates": int(self.num_states),
            "isEmpty": bool(self.is_empty),
            "isSymmetric": bool(self.is_symmetric),
            "numElements": int(self.num_elements),
            "numDiagonalElements": int(self.num_diagonal),
            "numUniqueValues": int(self.num_unique),
        }

        # Extract items from the SymPy sparse matrix
        dok = self.matrix.todok()
        num_elements = len(dok)
        coords = np.fromiter(dok.keys(), dtype=np.dtype((int, 2)), count=num_elements)
        elements = np.fromiter(dok.values(), dtype=object, count=num_elements)
        rows = coords[:, 0]
        columns = coords[:, 1]

        # Remove upper triangle of a symmetric matrix
        if self.is_symmetric:
            mask = rows >= columns
            rows = rows[mask]
            columns = columns[mask]
            elements = elements[mask]

        # Sort the unique element values
        values = sorted(set(elements), key=sp.default_sort_key)
        lookup = {val: i for i, val in enumerate(values)}
        elements = np.array([lookup[x] for x in elements])

        # Assert canonical order of row and column indices
        is_sorted = is_sorted_lex(rows, columns)
        if not is_sorted:
            sort_idx = np.lexsort((columns, rows))
            rows = rows[sort_idx]
            columns = columns[sort_idx]
            elements = elements[sort_idx]

        # Dictionary of sparse matrix elements
        matrix_dict = dict(info_meta)
        if self.num_elements > 0:
            matrix_dict |= encode_uint_array(rows, "rows")
            matrix_dict |= encode_uint_array(columns, "columns")
            matrix_dict |= encode_uint_array(elements, "elements")
            matrix_dict |= RationalRadicalList(values).as_dict()

        # Return dictionaries
        return matrix_dict, info_meta

    @staticmethod
    def hash_data(hasher, matrix_dict):
        """ Update hasher with matrix_dict. """

        for key in sorted(matrix_dict.keys()):
            value = matrix_dict[key]
            hasher.update(key.encode('utf-8'))
            if isinstance(value, np.ndarray):
                hasher.update(value.tobytes())
            else:
                hasher.update(str(value).encode('utf-8'))

    def collapse(self, indices, space, subspace):
        """ Return a new SymMatrix object containing only selected elements. """

        # Method is not implemented for transformation matrices yet
        if not (self.row_space == self.col_space == space):
            raise NotImplemented

        # Pick the selected matrix elements and return a new SymMatrix object
        matrix = self.matrix.extract(indices, indices)
        return self.from_matrix(subspace, subspace, matrix)

    @classmethod
    def reduced(cls, components, J):
        """ Return a SymMatrix object containing the reduced matrix elements of a tensor operator based its given
        ordered component matrices. """

        def sym3j(row, col, k, J):
            Ja, Jb = J[row], J[col]
            q = Ja - Jb
            assert -k <= q <= k
            factor = wigner_3j(Ja, k, Jb, -Ja, q, Jb)
            assert factor != 0
            return factor

        # Sanity checks
        assert all(isinstance(matrix, cls) for matrix in components)
        assert len(set(matrix.num_states for matrix in components)) == 1
        assert set(matrix.row_space for matrix in components) == {"SLJ"}
        assert set(matrix.col_space for matrix in components) == {"SLJ"}

        # Rank of the tensor operator
        assert len(components) % 2 == 1
        k = (len(components) - 1) // 2

        # Element-wise sum, using the fact that only one component is non-zero for each matrix element
        matrix = reduce(operator.add, [m.matrix for m in components])

        # Convert sum of components to matrix of reduced elements using the Wigner-Eckart theorem
        for (row, col), value in matrix.todok().items():
            matrix[row, col] = value / sym3j(row, col, k, J)
        return cls.from_matrix("SLJ", "SLJ", matrix)

    def array(self, dtype):
        """ Return matrix as numpy array. """

        return np.array(self.matrix.evalf()).astype(dtype)
