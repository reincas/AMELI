##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the class DataType as an abstraction
# layer to allow the same application code to act on symbolic and
# floating point values.
#
# The module also provides the classes SymMatrix and NumMatrix, which
# represent matrices of spherical tensor operators in a given state
# space. The former contains symbolic SymPy elements, the later numpy
# floating point elements.
#
##########################################################################

import math
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_3j

from .uintarray import encode_uint_array, decode_uint_array
from .symroots import RationalRadicalList


###########################################################################
# DataType interface class
###########################################################################

class DataType:
    """ Common abstraction class for symbolic (SymPy) and floating point (numpy) operations. """

    def __init__(self, name):
        """ Initialize a DataType object based on its name, which can be one of 'symbolic', 'float16', 'float32',
        'float64', or whatever floating point data type numpy supports. """

        # Store name of dtype
        self.name = name

        # Store flag to distinguish between symbolic and floating point operations
        self.is_symbolic = self.name == "symbolic"

        # Store SymPy or numpy data type object
        self.dtype = sp.Expr if self.is_symbolic else getattr(np, name)

        # Initialize basic symbolic attributes: Machine precision eps, numeric object of value 0 and numeric
        # object of value 1
        if self.is_symbolic:
            self.eps = 0
            self.zero = sp.S(0)
            self.one = sp.S(1)

        # Initialize basic floating point attributes
        else:
            assert np.issubdtype(self.dtype, np.floating)
            self.eps = np.finfo(self.dtype).eps
            self.zero = self.dtype(0)
            self.one = self.dtype(1)

    def __call__(self, value):
        """ Return given value as numeric object with self.dtype as data type. """

        if self.is_symbolic:
            return sp.S(value)
        return self.dtype(value)

    def sqrt(self, value):
        """ Return symbolic or floating point square root of the given value. """

        if self.is_symbolic:
            return sp.sqrt(value)
        return math.sqrt(value)

    def factorial(self, value):
        """ Return symbolic or floating point factorial of the given value. """

        if self.is_symbolic:
            return sp.factorial(value)
        return math.factorial(value)

    @lru_cache(maxsize=100000)
    def sym3j(self, j1, j2, j3, m1, m2, m3):
        """ Return symbolic or floating point result of the Wigner 3j-symbol. The given arguments of the 3j-symbol
        must be integers or SymPy expressions. The symbol evaluation takes place symbolic with infinite precision.
        The results of this method are cached. """

        value = wigner_3j(j1, j2, j3, m1, m2, m3)
        if self.is_symbolic:
            return value
        return self.dtype(value)

    def is_zero(self, value, factor=1):
        """ Return True if the given symbolic value is zero, or the given floating point value is almost zero with
        an absolute tolerance of factor * eps. """

        if self.is_symbolic:
            return value == 0
        return abs(value) <= factor * self.eps

    def zeros(self, rows, cols):
        """ Return symbolic or floating point matrix of zeros with given number of rows and columns. """

        if self.is_symbolic:
            return sp.zeros(rows, cols)
        return np.zeros((rows, cols), dtype=self.dtype)

    def ones(self, rows, cols):
        """ Return symbolic or floating point matrix of ones with given number of rows and columns. """

        if self.is_symbolic:
            return sp.ones(rows, cols)
        return np.ones((rows, cols), dtype=self.dtype)

    def eye(self, size):
        """ Return symbolic or floating point quadratic identity matrix with given number of rows and columns. """

        if self.is_symbolic:
            return sp.eye(size)
        return np.eye(size, dtype=self.dtype)

    def transform(self, matrix, transform):
        """ Apply the given transformation to the given matrix and return the result. """

        if self.is_symbolic:
            return transform.T * matrix * transform
        return transform.T @ matrix @ transform

    def array(self, matrix):
        """ Transform the given list or sympy.Matrix object into a floating point array and return it. This method
        is only available for floating point DataType objects. """

        assert not self.is_symbolic, "Method not available for symbolic DataType!"

        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if isinstance(matrix, sp.Matrix):
            matrix = np.array(matrix.tolist())
        return matrix.astype(self.dtype)

    def state_matrix(self, row_space: str, col_space: str, is_symmetric: bool, num_states: int):
        """ Return an empty initialised SymMatrix or NumMatrix object. """

        if self.is_symbolic:
            return SymMatrix(self, row_space, col_space, is_symmetric, num_states)
        return NumMatrix(self, row_space, col_space, is_symmetric, num_states)

    def from_matrix(self, row_space: str, col_space: str, matrix: sp.Matrix | np.ndarray):
        """ Return a SymMatrix or NumMatrix object initialised from the given matrix object. """

        if self.is_symbolic:
            return SymMatrix.from_matrix(self, row_space, col_space, matrix)
        return NumMatrix.from_matrix(self, row_space, col_space, matrix)

    def from_meta(self, matrix_dict: dict, info_meta: dict):
        """ Return a SymMatrix or NumMatrix object initialised from the given data container dictionaries. """

        if self.is_symbolic:
            return SymMatrix.from_meta(self, matrix_dict, info_meta)
        return NumMatrix.from_meta(self, matrix_dict, info_meta)

    @property
    def matrix_desc(self):
        """ Return meta data description string of the state matrix. """
        if self.is_symbolic:
            return SymMatrix.meta_desc
        return NumMatrix.meta_desc


###########################################################################
# Symbolic state matrix class
###########################################################################
SYM_DESC = """
The HDF5 item '{matrix_hdf5}' contains the {matrix} elements as {dtype} values.
The matrix is stored in a compressed format, which takes advantage of the sparsity of the matrix and its small number
of unique values.
The vector dataset 'rows' contains state indices related to the list of states in the item '{row_hdf5}'.
The vector dataset 'columns' contains state indices related to the list of states in the item '{col_hdf5}'.
For each row and column there is a value index in the dataset 'elements' which refers to the unique values
contained in the vector datasets 'signs', 'numerators', and 'denominators'.
The value of each matrix element is related to its sign s, numerator n, and denominator d by
value = (-1)^s * sqrt(n/d).
If the numerator or the denominator contain very large integers, the respective vector is split bitwise in parts
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
    """ This class represents a sparse quadratic matrix. The list 'values' contains all unique non-zero element
    values. For each non-zero matrix element there is an element in the lists rows, columns, and elements. Each
    element in the list 'elements' is an index to the respective matrix element value in the list 'values'. """

    # Data type of the matrix
    dtype: DataType

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
    _matrix: sp.Matrix | None

    # Immutable matrix flag
    is_immutable: bool

    # Sparse matrix representation for data container export / import
    rows: list
    columns: list
    elements: list
    values: list

    # Description of meta data
    meta_desc = SYM_DESC

    def __init__(self, dtype: DataType, row_space: str, col_space: str, is_symmetric: bool, num_states: int):
        """ Initialize empty mutable SymMatrix object. """

        # Store data type
        assert dtype.is_symbolic
        self.dtype = dtype

        # Store number of electron states (matrix dimension)
        self.num_states = int(num_states)

        # Store names of row and column state spaces
        self.row_space = row_space
        self.col_space = col_space

        # Store symmetry flag
        self.is_symmetric = bool(is_symmetric)

        # Empty sparse matrix structure
        self.rows = []
        self.columns = []
        self.elements = []
        self.values = []

        # Empty matrix
        self._matrix = None
        self.is_immutable = False

    @property
    def matrix(self):
        assert self.is_immutable
        if self._matrix is None:
            self._matrix = self.dtype.zeros(self.num_states, self.num_states)
            if not self.is_empty:
                for row, col, element in zip(self.rows, self.columns, self.elements):
                    self._matrix[row, col] = self.values[element]
                    if self.is_symmetric:
                        self._matrix[col, row] = self.values[element]
        return self._matrix

    def __setitem__(self, index: tuple, value):
        """ Store matrix element in the matrix object and the sparse matrix structure. """

        # Matrix must not be immutable
        assert not self.is_immutable

        # Skip zero value
        if not value:
            return

        # Store row and column indices
        row, column = index
        if self.is_symmetric:
            assert row >= column
        self.rows.append(row)
        self.columns.append(column)

        # Get value index and store the value, if it is a new one
        try:
            element = self.values.index(value)
        except ValueError:
            self.values.append(value)
            element = len(self.values) - 1

        # Store index of the matrix element value
        self.elements.append(element)

    def make_immutable(self):
        """ Determine and store matrix info and make the matrix immutable. """

        # Store or check info flags
        assert self._matrix is None
        assert len(self.rows) == len(self.columns) == len(self.elements) >= len(self.values)
        self.is_empty = len(self.elements) == 0
        if self.is_symmetric:
            assert all(row >= col for row, col in zip(self.rows, self.columns))

        # Sort symbolic values
        indices = np.argsort(self.values)
        self.values = [self.values[i] for i in indices]
        inv_indices = np.argsort(indices)
        self.elements = [inv_indices[i] for i in self.elements]

        # Store matrix info
        self.num_diagonal = sum(1 for row, col in zip(self.rows, self.columns) if row == col)
        if self.is_symmetric:
            self.num_elements = 2 * len(self.elements) - self.num_diagonal
        else:
            self.num_elements = len(self.elements)
        self.num_unique = len(self.values)

        # Matrix is immutable now
        self.is_immutable = True

    @classmethod
    def from_matrix(cls, dtype, row_space, col_space, matrix):
        """ Return an immutable SymMatrix object from a matrix object. """

        # Initialize empty SymMatrix object
        num_states = matrix.shape[0]
        is_symmetric = matrix.is_symmetric()
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Build sparse matrix structure and matrix object from matrix
        assert isinstance(matrix, sp.Matrix)
        assert len(obj.rows) == len(obj.columns) == len(obj.elements) == len(obj.values) == 0
        if not matrix.is_zero_matrix:
            for row in range(obj.num_states):
                max_col = row + 1 if obj.is_symmetric else obj.num_states
                for col in range(max_col):
                    obj[row, col] = matrix[row, col]

        # Determine matrix info and make the SymMatrix object immutable
        obj.make_immutable()

        # Store matrix object
        obj._matrix = matrix.copy()

        # Return the SymMatrix object
        return obj

    @classmethod
    def from_meta(cls, dtype, matrix_dict: dict, info_meta: dict):
        """ Return an immutable SymMatrix object from the data container dictionaries. """

        # Sanity check for redundant information
        for key, value in info_meta.items():
            assert matrix_dict[key] == info_meta[key]

        # Initialize empty SymMatrix object
        row_space = info_meta["rowSpace"]
        col_space = info_meta["colSpace"]
        num_states = info_meta["numStates"]
        is_symmetric = info_meta["isSymmetric"]
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store matrix elements
        if not info_meta["isEmpty"]:

            # Extract sparse matrix elements
            rows = decode_uint_array(matrix_dict, "rows")
            columns = decode_uint_array(matrix_dict, "columns")
            elements = decode_uint_array(matrix_dict, "elements")
            values = [value.expr for value in RationalRadicalList.from_dict(matrix_dict).values]

            # Store sparse matrix elements in the matrix object
            for row, col, element in zip(rows, columns, elements):
                obj[row, col] = values[element]

        # Determine matrix info and make the SymMatrix object immutable
        obj.make_immutable()

        # Sanity checks
        assert obj.is_empty == info_meta["isEmpty"]
        assert obj.num_elements == info_meta["numElements"]
        assert obj.num_diagonal == info_meta["numDiagonalElements"]
        assert obj.num_unique == info_meta["numUniqueValues"]

        # Return SymMatrix object
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

        # Dictionary of sparse matrix elements
        matrix_dict = dict(info_meta)
        if self.num_elements > 0:
            matrix_dict |= encode_uint_array(self.rows, "rows")
            matrix_dict |= encode_uint_array(self.columns, "columns")
            matrix_dict |= encode_uint_array(self.elements, "elements")
            matrix_dict |= RationalRadicalList(self.values).as_dict()

        # Return dictionaries
        return matrix_dict, info_meta

    def collapse(self, indices, space, subspace):
        """ Return a new SymMatrix object containing only selected elements. """

        # Method is not implemented for transformation matrices yet
        if not (self.row_space == self.col_space == space):
            raise NotImplemented

        # Initialize empty SymMatrix object
        num_states = len(indices)
        obj = SymMatrix(self.dtype, subspace, subspace, self.is_symmetric, num_states)

        # Store selected matrix elements in the matrix object
        if not self.is_empty:
            for row, col, element in zip(self.rows, self.columns, self.elements):
                if row in indices and col in indices:
                    i = indices.index(row)
                    j = indices.index(col)
                    obj[i, j] = self.values[element]

        # Determine matrix info and make the SymMatrix object immutable
        obj.make_immutable()

        # Return collapsed SymMatrix object
        return obj


###########################################################################
# Floating point state matrix class
###########################################################################

NUM_DESC = """
The HDF5 item '{matrix_hdf5}' contains the {matrix} elements as {dtype} values.
The dataset 'matrix' contains the full matrix.
Row indices are related to the list of states in the item '{row_hdf5}'.
Column indices are related to the list of states in the item '{col_hdf5}'.
The size of the quadratic matrix is given by the square of the attribute 'numStates'.
The attribute 'isSymmetric' is True for a symmetric matrix. 
The attribute 'numElements' contains the approximate number of non-zero matrix elements.
The attribute 'numDiagonalElements' contains the approximate number of non-zero matrix elements on the matrix diagonal.
The attribute 'numUniqueValues' contains the approximate number of different non-zero matrix element values.
The approximations are based on a tolerance range of a small multiple of the machine precision eps of the {dtype}
data type.
"""


class NumMatrix:
    """ This class represents a sparse quadratic matrix. The list 'values' contains all unique non-zero element
    values. For each non-zero matrix element there is an element in the lists rows, columns, and elements. Each
    element in the list 'elements' is an index to the respective matrix element value in the list 'values'. """

    # Data type of the matrix
    dtype: DataType

    # State space strings
    row_space: str
    col_space: str

    # Number of electrons (matrix size is num_states * num_states)
    num_states: int

    # Empty matrix flag
    is_empty: bool

    # Symmetric matrix flag
    is_symmetric: bool

    # Number of non-zero matrix elements in total (approximate)
    num_elements: int

    # Number of non-zero matrix elements on the matrix diagonal (approximate)
    num_diagonal: int

    # Number of unique non-zero matrix elements (approximate)
    num_unique: int

    # Matrix object
    matrix: np.ndarray

    # Immutable matrix flag
    is_immutable: bool

    # Description of meta data
    meta_desc = NUM_DESC

    def __init__(self, dtype: DataType, row_space: str, col_space: str, is_symmetric: bool, num_states: int):
        """ Initialize empty mutable NumMatrix object. """

        # Store data type
        assert not dtype.is_symbolic
        self.dtype = dtype

        # Store number of electron states (matrix dimension)
        self.num_states = int(num_states)

        # Store names of row and column state spaces
        self.row_space = row_space
        self.col_space = col_space

        # Store symmetry flag
        self.is_symmetric = bool(is_symmetric)

        # Empty matrix
        self.matrix = self.dtype.zeros(self.num_states, self.num_states)
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
        assert len(self.matrix.shape) == 2
        assert self.matrix.shape[0] == self.matrix.shape[1] == self.num_states
        self.is_empty = bool(not self.matrix.any())
        assert self.is_symmetric == bool(np.all(self.matrix.T == self.matrix))

        # Collect approximate non-zero elements
        rtol = atol = 10 * self.dtype.eps
        elements = self.matrix.ravel()[np.abs(self.matrix.ravel()) > atol]
        diagonal = self.matrix.diagonal()[np.abs(self.matrix.diagonal()) > atol]
        is_unique = ~np.isclose(elements[:-1], elements[1:], rtol=rtol, atol=atol)

        # Store approximate numbers of non-zero elements
        self.num_elements = len(elements)
        self.num_diagonal = len(diagonal)
        self.num_unique = 0 if len(elements) == 0 else int(sum(is_unique)) + 1

        # Matrix is immutable now
        self.is_immutable = True

    @classmethod
    def from_matrix(cls, dtype, row_space, col_space, matrix):
        """ Create an immutable NumMatrix object from a matrix object. """

        # Initialize empty NumMatrix object
        num_states = matrix.shape[0]
        is_symmetric = np.all(matrix.T == matrix)
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store matrix
        assert matrix.dtype == dtype.dtype
        assert isinstance(obj.matrix, np.ndarray)
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == matrix.shape[1]
        obj.matrix = matrix

        # Determine matrix info and make the NumMatrix object immutable
        obj.make_immutable()

        # Return the NumMatrix object
        return obj

    @classmethod
    def from_meta(cls, dtype, matrix_dict: dict, info_meta: dict):
        """ Return an immutable NumMatrix object from the data container dictionaries. """

        # Sanity check for redundant information
        for key, value in info_meta.items():
            assert matrix_dict[key] == info_meta[key]

        # Initialize empty NumMatrix object
        row_space = info_meta["rowSpace"]
        col_space = info_meta["colSpace"]
        num_states = info_meta["numStates"]
        is_symmetric = info_meta["isSymmetric"]
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store matrix object
        obj.matrix = matrix_dict["matrix"]

        # Determine matrix info and make the NumMatrix object immutable
        obj.make_immutable()

        # Sanity checks
        assert obj.is_empty == info_meta["isEmpty"]
        assert obj.num_elements == info_meta["numElements"]
        assert obj.num_diagonal == info_meta["numDiagonalElements"]
        assert obj.num_unique == info_meta["numUniqueValues"]

        # Return NumMatrix object
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

        # Dictionary of matrix elements
        matrix_dict = dict(info_meta)
        matrix_dict["matrix"] = self.matrix

        # Return dictionaries
        return matrix_dict, info_meta

    def collapse(self, indices, space, subspace):
        """ Return a new NumMatrix object containing only selected elements. """

        # Method is not implemented for transformation matrices yet
        if not (self.row_space == self.col_space == space):
            raise NotImplemented

        # Pick the selected matrix elements and return a new NumMatrix object
        matrix = self.matrix[np.ix_(indices, indices)]
        assert len(matrix.shape) == 2, f"{self.matrix.shape}, {matrix.shape}, {indices}"
        return self.from_matrix(self.dtype, subspace, subspace, matrix)
