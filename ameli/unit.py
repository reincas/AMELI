##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Unit, which represents the matrix of
# a spherical mixed unit tensor operator in the product state space.
#
##########################################################################
import hashlib
import logging
import time
from abc import ABC, abstractmethod
import sympy as sp

from . import desc_format, sym3j
from .states import space_registry
from .sparse import SymMatrix
from .config import ConfigInfo, Config
from .product import Product
from .vault import AMELI_VERSION, VersionError, Vault

__version__ = "1.0.0"
logger = logging.getLogger("unit")


###########################################################################
# Unit generator objects
###########################################################################

class BaseUnit(ABC):
    """ Base factory class for the calculation of the state matrix of a tensor operator. Each matrix element
    is a sum of matrix elements of an elementary tensor operator acting on tensor_size electrons. The value
    of this elementary matrix element is calculated in the method calc_element() of the subclass. The method
    as_dict() returns a dictionary with all matrix elements in a compact form ready for storage in a data
    container, which takes advantage of the sparsity and low number of different values. """

    def __init__(self, product, tensor_size, parameters, symmetric, factor, template):
        """ Calculate all matrix elements and store them in a SymMatrix object. """

        # Product object used to get all potentially non-zero matrix elements
        self.product = product
        self.num_subshells = self.product.config.num_subshells

        # Number of electrons, the operator is acting on
        self.tensor_size = int(tensor_size)
        assert self.tensor_size == self.product.tensor_size

        # Common factor of all matrix elements prepared by the subclass
        factor = factor / sp.factorial(self.tensor_size)

        # Generate string expression of the elementary tensor
        self.parameters = parameters
        self.expression = template.format(**parameters)
        self.template = template.format(**{key: key for key in parameters})

        # Sanity check
        assert isinstance(symmetric, bool)

        # Shortcut: use given matrix
        if isinstance(product, SymMatrix):
            self.matrix = product
            assert self.matrix.row_space == "Product"
            assert self.matrix.col_space == "Product"
            assert self.matrix.is_symmetric == symmetric
            assert self.matrix.num_states == self.product.num_states
            return

        # Calculate all non-zero matrix elements
        self.matrix = SymMatrix("Product", "Product", symmetric, self.product.num_states)
        for initial, final, elements in self.product.matrix_elements():

            # Convert numpy integer objects into Python int
            initial = int(initial)
            final = int(final)

            # Sanity check
            assert initial <= final

            # Diagonal element or element of a symmetric matrix
            if initial == final or symmetric:
                value = sp.S(0)
                for electrons, sign in elements:
                    value += self.element(electrons, sign, swapped=True)
                self.matrix[final, initial] = value * factor

            # Element of an asymmetric matrix
            # Note: The generator elements can be used only once!
            else:
                value_ll = sp.S(0)
                value_ur = sp.S(0)
                for electrons, sign in elements:
                    value_ll += self.element(electrons, sign, swapped=True)
                    value_ur += self.element(electrons, sign, swapped=False)
                self.matrix[final, initial] = value_ll * factor
                self.matrix[initial, final] = value_ur * factor

    def element(self, electrons, sign, swapped):
        """ Return the SymPy value of a matrix element of the elementary operator with global sign and optional
        swapping of initial and final electrons. """

        if not swapped:
            final, initial = electrons[:self.tensor_size], electrons[self.tensor_size:]
        else:
            final, initial = electrons[self.tensor_size:], electrons[:self.tensor_size]
        element = self.calc_element(*final, *initial)
        if sign:
            return -element
        return element

    @abstractmethod
    def calc_element(self, *electrons):
        """ Return the SymPy value of a matrix element of the elementary operator for the given initial and
        final electrons. This method is implemented in the subclasses. """
        pass


class Unit_UT(BaseUnit):
    """ This class provides component q of an elementary mixed one-electron tensor operator Q of rank k. The
    operator Q is the product of a unit tensor operator u^(k1) of rank k1 in the orbital space and a unit tensor
    operator t^(k2) of rank k2 in the spin space: Q^(k)_q = {u^(k1) x t^(k2)}^(k)_q """

    def __init__(self, product, k1, k2, k, q):
        """ Calculate and store all matrix elements. """

        # Tensor ranks and component q
        self.k1 = sp.S(k1)
        self.k2 = sp.S(k2)
        self.k = sp.S(k)
        self.q = sp.S(q)

        # Calculate all matrix elements
        tensor_size = 1
        parameters = {"k1": k1, "k2": k2, "k": k, "q": q}
        symmetric = q == 0
        factor = sp.sqrt(2 * self.k + 1)
        template = "{{u({k1}) x t({k2})}}({k})_{q}"
        super().__init__(product, tensor_size, parameters, symmetric, factor, template)

    def calc_element(self, a, b):
        """ Return the SymPy value of the one-electron matrix element <a|Q|b> of component q of the rank-k
        operator Q with the final electron state <a| and the initial electron state |b>. """

        # Shortcut for parameters known to result in zero
        if self.num_subshells > 1:
            if a.l != b.l or a.s != b.s:
                return 0
        if a.ml - b.ml + a.ms - b.ms != self.q:
            return 0

        # Avoid unnecessary evaluations if a factor is zero
        def factors():
            yield sym3j(self.k1, self.k, self.k2, a.ml - b.ml, -self.q, a.ms - b.ms)
            yield sym3j(a.l, self.k1, b.l, -a.ml, a.ml - b.ml, b.ml)
            yield sym3j(a.s, self.k2, b.s, -a.ms, a.ms - b.ms, b.ms)

        # Calculate product of 3j-symbols
        element = sp.S(1)
        for factor in factors():
            element *= factor
            if not element:
                return 0

        # Return value with sign
        if (self.k + a.l + a.s - b.ml - b.ms) % 2:
            return -element
        return element


class Unit_UTUT(BaseUnit):
    """ This class provides component q of an elementary mixed two-electron tensor operator Q of rank k. The
    operator Q is the product of a unit tensor operator {u_1^(k1) x t_1^(k2)}^(k_12) acting on one electron and
    another tensor operator {u_2^(k3) x t_2^(k4)}^(k_34) acting on a second electron. In total:
    Q^(k)_q = {{u_1^(k1) x t_1^(k2)}^(k_12) x {u_2^(k3) x t_2^(k4)}^(k_34)}^(k)_q """

    def __init__(self, product, k1, k2, k12, k3, k4, k34, k, q):
        """ Calculate and store all matrix elements. """

        # Tensor ranks and component q
        self.k1 = sp.S(k1)
        self.k2 = sp.S(k2)
        self.k12 = sp.S(k12)
        self.k3 = sp.S(k3)
        self.k4 = sp.S(k4)
        self.k34 = sp.S(k34)
        self.k = sp.S(k)
        self.q = sp.S(q)

        # Calculate all matrix elements
        tensor_size = 2
        parameters = {"k1": k1, "k2": k2, "k12": k12, "k3": k3, "k4": k4, "k34": k34, "k": k, "q": q}
        symmetric = q == 0
        factor = sp.sqrt((2 * self.k + 1) * (2 * self.k12 + 1) * (2 * self.k34 + 1))
        template = "{{{{u1({k1}) x t1({k2})}}({k12}) x {{u2({k3}) x t2({k4})}}({k34})}}({k})_{q}"
        super().__init__(product, tensor_size, parameters, symmetric, factor, template)

    def calc_element(self, a, b, c, d):
        """ Return the SymPy value of the two-electron matrix element <a,b|Q|c,d> of component q of the rank-k
        operator Q with the final electron state <a,b| and the initial electron state |c,d>. """

        # Shortcut for parameters known to result in zero
        if self.num_subshells > 1:
            if a.l != c.l or a.s != c.s:
                return 0
            if b.l != d.l or b.s != d.s:
                return 0
        if a.ml + b.ml - c.ml - d.ml + a.ms + b.ms - c.ms - d.ms != self.q:
            return 0

        # Avoid unnecessary evaluations if a factor is zero
        def factors():
            yield sym3j(self.k12, self.k, self.k34, a.ml - c.ml + a.ms - c.ms, -self.q, b.ml - d.ml + b.ms - d.ms)
            yield sym3j(self.k1, self.k12, self.k2, a.ml - c.ml, -a.ml + c.ml - a.ms + c.ms, a.ms - c.ms)
            yield sym3j(a.l, self.k1, c.l, -a.ml, a.ml - c.ml, c.ml)
            yield sym3j(a.s, self.k2, c.s, -a.ms, a.ms - c.ms, c.ms)
            yield sym3j(self.k3, self.k34, self.k4, b.ml - d.ml, -b.ml + d.ml - b.ms + d.ms, b.ms - d.ms)
            yield sym3j(b.l, self.k3, d.l, -b.ml, b.ml - d.ml, d.ml)
            yield sym3j(b.s, self.k4, d.s, -b.ms, b.ms - d.ms, d.ms)

        # Calculate product of 3j-symbols
        element = sp.S(1)
        for factor in factors():
            element *= factor
            if not element:
                return 0

        # Return value with sign
        if (self.k + self.q + self.k12 + self.k34 + a.l + b.l + a.s + b.s - c.ml - d.ml - c.ms - d.ms) % 2:
            return -element
        return element


class Unit_UUU(BaseUnit):
    """ This class provides the elementary three-electron triple tensor operator scalar product Q of the unit
    tensor operators u_1^(k1), u_2^(k2), and u_3^(k3) of ranks k1, k2, and k3 in the orbital space:
    Q = (u_1^(k1) · u_2^(k2) · u_3^(k3)). """

    def __init__(self, product, k1, k2, k3):
        """ Calculate and store all matrix elements. """

        # Tensor ranks
        self.k1 = sp.S(k1)
        self.k2 = sp.S(k2)
        self.k3 = sp.S(k3)

        # Calculate all matrix elements
        tensor_size = 3
        parameters = {"k1": k1, "k2": k2, "k3": k3}
        symmetric = True
        factor = sp.S(1)
        template = "(u1({k1}) · u2({k2}) · u3({k3}))"
        super().__init__(product, tensor_size, parameters, symmetric, factor, template)

    def calc_element(self, a, b, c, d, e, f):
        """ Return the SymPy value of the three-electron matrix element <a,b,c|Q|d,e,f> of the triple scalar
         product Q with the final electron state <a,b,c| and the initial electron state |d,e,f>. """

        # Shortcut for parameters known to result in zero
        if self.num_subshells > 1:
            if a.l != d.l or a.s != d.s:
                return 0
            if b.l != e.l or b.s != e.s:
                return 0
            if c.l != f.l or c.s != f.s:
                return 0
        if a.ms != d.ms or b.ms != e.ms or c.ms != f.ms:
            return 0

        # Avoid unnecessary evaluations if a factor is zero
        def factors():
            yield sym3j(self.k1, self.k2, self.k3, a.ml - d.ml, b.ml - e.ml, c.ml - f.ml)
            yield sym3j(a.l, self.k1, d.l, -a.ml, a.ml - d.ml, d.ml)
            yield sym3j(b.l, self.k2, e.l, -b.ml, b.ml - e.ml, e.ml)
            yield sym3j(c.l, self.k3, f.l, -c.ml, c.ml - f.ml, f.ml)

        # Calculate product of 3j-symbols
        element = sp.S(1)
        for factor in factors():
            element *= factor
            if not element:
                return 0

        # Return value with sign
        if (a.l + b.l + c.l - a.ml - b.ml - c.ml) % 2:
            return -element
        return element


# Dictionary mapping names to unit matrix classes and tensor sizes
MATRIX = {
    "UT": (Unit_UT, 1),
    "UTUT": (Unit_UTUT, 2),
    "UUU": (Unit_UUU, 3),
}

###########################################################################
# Unit class
###########################################################################

TITLE = "Spherical unit tensor operator product state matrix elements"

DESCRIPTION = """
This container stores the product state matrix elements of a mixed spherical unit tensor operator in the given
many-electron configuration.
<br> {states_desc}
<br> {matrix_desc}
"""


class Unit(Vault):
    """ Class of the product state matrix of a mixed unit spherical tensor operator. It provides the SymPy matrix in
    the attribute 'matrix'. """

    def __init__(self, config_name, name):
        """ Initialize the spherical unit tensor operator matrix. """

        # Configuration string
        self.config_name = config_name

        # Matrix name
        self.name = name

        # Load or generate data container
        self.file = self.get_path(config_name, name)
        dc = self.load_container(self.file, __version__)

        # Extract UUID and code version from the container
        meta = dc["data/unit.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Sanity check for data type
        assert meta["dataType"] == "symbolic"

        # Characteristics of the unit tensor operator
        assert meta["name"] == self.name
        self.tensor_size = meta["numTensorElectrons"]
        self.parameters = meta["tensorParameters"]
        self.expression = meta["tensorExpression"]
        self.template = meta["tensorTemplate"]

        # Extract electron configuration from the container
        self.config = ConfigInfo.from_meta(meta["config"])
        assert self.config.name == config_name

        # Extract product states from the container
        self.states = space_registry["Product"].from_meta(dc["data/states.hdf5"], meta["states"])

        # Extract unit matrix
        self.info = SymMatrix.from_meta(dc["data/matrix.hdf5"], meta["matrix"])
        self.matrix = self.info.matrix
        assert self.info.row_space == self.states.state_space
        assert self.info.col_space == self.states.state_space

    def generate_container(self, dc=None):
        """ Generate the matrix of the unit tensor operator and store it in a data container file. Update function: Use
        the matrix from the given data container if provided. """

        logger.info(f"Generating {self.config_name} unit matrix {self.name}")
        t = time.time()

        # Initialize the data hasher
        hasher = hashlib.sha256()

        # Get electron configuration
        config = Config(self.config_name)
        config_meta = config.info.as_meta()

        # Get product states
        states_dict, states_meta = config.states.as_meta(hasher)

        # Calculate elementary matrix elements
        key, parameters = self.name.split("/")
        cls, tensor_size = MATRIX[key]
        parameters = tuple(map(int, parameters.split(",")))
        if dc:
            product = SymMatrix.from_meta(dc["data/matrix.hdf5"], dc["data/unit.json"]["matrix"]).matrix
        else:
            product = Product(self.config_name, tensor_size)
        unit = cls(product, *parameters)
        assert unit.tensor_size == tensor_size

        # Get matrix data dictionaries
        matrix_dict, matrix_meta = unit.matrix.as_meta(hasher)
        logger.debug(f" {self.config_name} | Finished unit matrix {self.name}")

        # Generate data hash
        data_hash = hasher.hexdigest()
        if dc and data_hash == dc["content.json"]["sha256Data"]:
            raise VersionError

        # Prepare container description string
        kwargs = {
            "states": "states",
            "states_hdf5": "states.hdf5",
            "matrix_hdf5": "matrix.hdf5",
            "matrix": "unit tensor matrix",
            "row_hdf5": "states.hdf5",
            "col_hdf5": "states.hdf5",
            "json": "unit.json",
        }
        kwargs["states_desc"] = desc_format(config.states_desc, kwargs)
        kwargs["matrix_desc"] = desc_format(SymMatrix.meta_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Initialise container structure
        items = {
            "content.json": {
                "containerType": {"name": "ameliUnit"},
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
            "data/unit.json": {
                "version": __version__,
                "dataType": "symbolic",
                "name": self.name,
                "config": config_meta,
                "states": states_meta,
                "matrix": matrix_meta,
                "numTensorElectrons": unit.tensor_size,
                "tensorParameters": unit.parameters,
                "tensorExpression": unit.expression,
                "tensorTemplate": unit.template,
            },
            "data/states.hdf5": states_dict,
            "data/matrix.hdf5": matrix_dict,
        }

        # Create the data container and store it in a file
        self.write(self.file, items)
        t = time.time() - t
        logger.info(f"Stored {self.config_name} unit matrix {self.name} ({t:.1f} seconds) -> {self.file}")

    @staticmethod
    def get_path(config_name, name):
        """ Return data container file name. """

        key, parameters = name.split("/")
        parameters = "_".join(parameters.split(","))
        return f"{config_name}/unit/{key}_{parameters}.zdc"
