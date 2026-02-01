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
import operator
from functools import lru_cache, reduce

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
            return sp.SparseMatrix(rows, cols, {})
        return np.zeros((rows, cols), dtype=self.dtype)

    def eye(self, size):
        """ Return symbolic or floating point quadratic identity matrix with given number of rows and columns. """

        if self.is_symbolic:
            return sp.SparseMatrix.eye(size)
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

    def from_matrix(self, row_space: str, col_space: str, matrix: sp.SparseMatrix | np.ndarray):
        """ Return a SymMatrix or NumMatrix object initialised from the given matrix object. """

        if self.is_symbolic:
            return SymMatrix.from_matrix(self, row_space, col_space, matrix)
        return NumMatrix.from_matrix(self, row_space, col_space, matrix)

    def from_meta(self, matrix_dict: dict, info_meta: dict):
        """ Return a SymMatrix or NumMatrix object initialised from the given data container dictionaries. """

        if self.is_symbolic:
            return SymMatrix.from_meta(self, matrix_dict, info_meta)
        return NumMatrix.from_meta(self, matrix_dict, info_meta)

    def reduced(self, components: list, J: list):
        """ Return a SymMatrix or NumMatrix object containing the reduced matrix elements of a tensor operator based
        on its given ordered component matrices. """

        if self.is_symbolic:
            return SymMatrix.reduced(components, J)
        return NumMatrix.reduced(components, J)

    @property
    def matrix_desc(self):
        """ Return meta data description string of the state matrix. """
        if self.is_symbolic:
            return SymMatrix.meta_desc
        return NumMatrix.meta_desc


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
    matrix: sp.SparseMatrix

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
        assert self.is_symmetric == self.matrix.is_symmetric()

        # Statistics of non-zero elements
        self.num_elements = self.matrix.nnz()
        self.num_diagonal = self.matrix.diagonal().nnz()
        self.num_unique = len(set(self.matrix.todok().values()))

        # Matrix is immutable now
        self.is_immutable = True

    @classmethod
    def from_matrix(cls, dtype, row_space, col_space, matrix):
        """ Return an immutable SymMatrix object from a matrix object. """

        # Initialize empty matrix object
        num_states = matrix.shape[0]
        is_symmetric = matrix.is_symmetric()
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store sparse SymPy matrix
        assert isinstance(matrix, sp.SparseMatrix)
        obj.matrix = matrix

        # Determine matrix info and make the matrix object immutable
        obj.make_immutable()

        # Return the matrix object
        return obj

    @classmethod
    def from_meta(cls, dtype, matrix_dict: dict, info_meta: dict):
        """ Return an immutable SymMatrix object from the data container dictionaries. """

        # Sanity check for redundant information
        for key, value in info_meta.items():
            assert matrix_dict[key] == info_meta[key]

        # Initialize empty matrix object
        row_space = info_meta["rowSpace"]
        col_space = info_meta["colSpace"]
        num_states = info_meta["numStates"]
        is_symmetric = info_meta["isSymmetric"]
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

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

    def collapse(self, indices, space, subspace):
        """ Return a new SymMatrix object containing only selected elements. """

        # Method is not implemented for transformation matrices yet
        if not (self.row_space == self.col_space == space):
            raise NotImplemented

        # Pick the selected matrix elements and return a new SymMatrix object
        matrix = self.matrix.extract(indices, indices)
        return self.from_matrix(self.dtype, subspace, subspace, matrix)

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
        assert set(matrix.dtype.name for matrix in components) == {"symbolic"}
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
        dtype = components[0].dtype
        return cls.from_matrix(dtype, "SLJ", "SLJ", matrix)

    def array(self, dtype):
        """ Return matrix as numpy array. """

        return np.array(self.matrix.evalf()).astype(dtype)

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

        # Initialize empty matrix object
        num_states = matrix.shape[0]
        is_symmetric = np.all(matrix.T == matrix)
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store numpy matrix object
        assert matrix.dtype == dtype.dtype
        assert isinstance(obj.matrix, np.ndarray)
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == matrix.shape[1]
        obj.matrix = matrix

        # Determine matrix info and make the matrix object immutable
        obj.make_immutable()

        # Return the matrix object
        return obj

    @classmethod
    def from_meta(cls, dtype, matrix_dict: dict, info_meta: dict):
        """ Return an immutable NumMatrix object from the data container dictionaries. """

        # Sanity check for redundant information
        for key, value in info_meta.items():
            assert matrix_dict[key] == info_meta[key]

        # Initialize empty matrix object
        row_space = info_meta["rowSpace"]
        col_space = info_meta["colSpace"]
        num_states = info_meta["numStates"]
        is_symmetric = info_meta["isSymmetric"]
        obj = cls(dtype, row_space, col_space, is_symmetric, num_states)

        # Store matrix object
        obj.matrix = matrix_dict["matrix"]

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

        # Pick the selected matrix elements and return a new matrix object
        matrix = self.matrix[np.ix_(indices, indices)]
        assert len(matrix.shape) == 2, f"{self.matrix.shape}, {matrix.shape}, {indices}"
        return self.from_matrix(self.dtype, subspace, subspace, matrix)

    @classmethod
    def reduced(cls, components, J):
        """ Return a NumMatrix object containing the reduced matrix elements of a tensor operator based on its given
        ordered component matrices. """

        # Sanity checks
        assert all(isinstance(matrix, cls) for matrix in components)
        assert len(set(matrix.num_states for matrix in components)) == 1
        assert len(set(matrix.dtype.name for matrix in components)) == 1
        assert set(matrix.row_space for matrix in components) == {"SLJ"}
        assert set(matrix.col_space for matrix in components) == {"SLJ"}

        # Rank of the tensor operator
        assert len(components) % 2 == 1
        k = (len(components) - 1) // 2

        # Element-wise sum, using the fact that only one component is non-zero for each matrix element
        dtype = components[0].dtype
        matrix = np.sum([matrix.matrix for matrix in components], axis=0)

        # Convert sum of components to matrix of reduced elements using the Wigner-Eckart theorem
        num_states = matrix.shape[0]
        for i in range(num_states):
            for j in range(num_states):
                Ja = J[i]
                Jb = J[j]
                q = Ja - Jb
                if q < -k or q > k:
                    assert np.abs(matrix[i, j]) < 1000 * dtype.eps
                    continue
                factor = wigner_3j(Ja, k, Jb, -Ja, q, Jb)
                if factor == 0:
                    assert np.abs(matrix[i, j]) < 1000 * dtype.eps
                    continue
                matrix[i, j] /= dtype(factor)

        # Return matrix of reduced elements
        return cls.from_matrix(dtype, "SLJ", "SLJ", matrix)
