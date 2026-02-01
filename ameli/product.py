##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Product, which holds support data for
# the calculation of product state matrix elements of spherical tensor
# operators.
#
##########################################################################

import logging
import math
import os
import tempfile
import time
from itertools import combinations
import h5py
import numpy as np

from . import desc_format
from .config import ConfigInfo, Config
from .states import space_registry
from .uintarray import get_dtype
from .vault import container_vault

__version__ = "1.0.0"
logger = logging.getLogger("product")


###########################################################################
# Product generator objects
###########################################################################

class ProductState:
    """ Support class of an electron product state. The object represents an ordered tuple of electrons. It supports
    a fast decision if two ProductState objects differ in exactly num_diff electrons from the intersection of the
    sets of potential same electrons delivered by the methods same_electrons(num_diff) of both objects. The respective
    tuple of same electrons can then be used to get the differing electrons and the reorder-sign using the method
    other_electrons(same_electrons) of both ProductState objects. """

    def __init__(self, electrons):
        """ Store the tuple of electrons and initialize the data structures for a fast determination of matching
        electrons in two ProductState objects. """

        # Tuple of electrons (indices into the electron pool of the configuration)
        self.electrons = tuple(electrons)
        self.num_electrons = len(self.electrons)

        # Store the {same: (other, sign)}-dictionaries for up to max_diff removed (other) electrons
        self._matches_ = []

    def generate_match(self, num_diff):
        """ This method generates data structures which support a fast decision if two ProductState objects differ
        in exactly num_diff electrons. It removes every num_diff-combination of electrons from this state and stores
        the results a dictionary which is returned. Every key of this dictionary is an ordered tuple of remaining
        electrons, the value is a 2-tuple containing the ordered tuple of removed electrons and the reorder-sign.
        This sign is based on the number of neighbour swap operations required to move the removed electrons to
        the end of the ordered sequence of state electrons. """

        # Initialize the {same: (other, sign)}-dictionary
        matches = {}

        # Special case of no removed electrons
        if num_diff == 0:
            same_electrons = self.electrons
            other_electrons = ()
            sign = 0
            matches[same_electrons] = (other_electrons, sign)

        # Remove all combinations of num_diff electrons from this state
        elif num_diff <= self.num_electrons:

            # Index sequence is required to keep track of the electron order and calculate the sign
            indices = range(self.num_electrons)

            # Select all combinations of removed (other) and remaining (same) electrons
            for selected in combinations(indices, num_diff):
                same_electrons = tuple(self.electrons[i] for i in indices if i not in selected)
                other_electrons = tuple(self.electrons[i] for i in selected)

                # Number of neighbour swap operations to shift the selected electrons to the end
                swaps = num_diff * self.num_electrons - sum(range(num_diff + 1)) - sum(selected)

                # Sign is 0 or 1 for an even or odd number of swaps
                sign = swaps % 2

                # Store results in the dictionary
                matches[same_electrons] = (other_electrons, sign)

        # Return the {same: (other, sign)}-dictionary for num_diff removed (other) electrons
        return matches

    def matches(self, num_diff):
        """ Generate and store all matching data structures upto num_diff. """

        assert num_diff >= 0
        while len(self._matches_) <= num_diff:
            self._matches_.append(self.generate_match(len(self._matches_)))
        return self._matches_[num_diff]

    def same_electrons(self, num_diff):
        """ Return the set of all ordered tuples containing the remaining electrons of this state when any combination
        of num_diff electrons is removed. """

        return set(self.matches(num_diff).keys())

    def other_electrons(self, same_electrons):
        """ Return the ordered tuple of electrons of this state missing in the tuple same_electrons and the sign of
        the neighbor swap operations required to move the removed electrons to the end of the ordered sequence of
        electrons of this state. """

        num_diff = self.num_electrons - len(same_electrons)
        return self.matches(num_diff)[same_electrons]


class ProductElement():
    """ Support class of a determinantal product state matrix element. The object represents two n-sequences of
    initial and final electrons sharing a certain number of common electrons (same_electrons). The method
    elementary(tensor_size) is used to deliver a tuple of tensor_size initial and final electrons together with
    the respective sign. Therefore, each tuple element contains the arguments required for the evaluation of
    an elementary operator acting on tensor_size electrons. The sum of these evaluations with sign results in
    a matrix element of the full tensor operator in a configuration of n electrons. """

    def __init__(self, same_electrons, initial_other_electrons, final_other_electrons, sign):
        """ Store equal (same) and different (other) electrons of the initial and final state as well as the
        combined sign of the states. """

        assert isinstance(same_electrons, tuple)
        assert isinstance(initial_other_electrons, tuple)
        assert isinstance(final_other_electrons, tuple)
        assert isinstance(sign, int)
        assert len(final_other_electrons) == len(initial_other_electrons)

        # Common and different electrons of initial and final state
        self.same_electrons = same_electrons
        self.initial_other_electrons = initial_other_electrons
        self.final_other_electrons = final_other_electrons

        # Global sign of the matrix element
        self.sign = sign % 2

        # Number of different electrons
        self.num_diff = len(self.initial_other_electrons)

        # Total number of electrons in each state
        self.num_electrons = len(self.same_electrons) + self.num_diff

    def iterate(self, size):
        """ Generate all initial and final electron sequences with the given number of electrons. Picks all
        possible combinations of same electrons and combines them with the fixed other electrons. """

        # Generator for all (size-num_diff)-combinations from the sequence of same electrons
        def generate(min_index, same):
            if len(same) + self.num_diff < size:
                for index in range(min_index, len(self.same_electrons)):
                    electron = self.same_electrons[index]
                    yield from generate(index + 1, same + (electron,))
            else:
                initial_electrons = same + self.initial_other_electrons
                final_electrons = same + self.final_other_electrons
                yield initial_electrons, final_electrons

        # Start the combination generator
        yield from generate(0, ())

    @staticmethod
    def determinant(electrons):
        """ Generate all permutations with sign from the given sequence of electrons for the construction of
        antisymmetric determinantal product states. """

        # Permutation generator. Requires a global sign value passed as one-element list by reference. An int sign
        # would be passed by value and local modifications would be lost in the recursive calls.
        def generate(k, electrons, sign):
            if k > 1:
                yield from generate(k - 1, electrons, sign)
                for i in range(k - 1):
                    if k % 2 == 0:
                        electrons[i], electrons[k - 1] = electrons[k - 1], electrons[i]
                    else:
                        electrons[0], electrons[k - 1] = electrons[k - 1], electrons[0]
                    sign[0] = (sign[0] + 1) % 2
                    yield from generate(k - 1, electrons, sign)
            else:
                yield tuple(electrons), sign[0] % 2

        # Start the permutation generator
        yield from generate(len(electrons), list(electrons), [0])

    def elementary(self, tensor_size, dtype):
        """ Return array of all combinations and permutations of tensor_size-sequences of initial and final
        electrons together with their antisymmetry sign for the calculation of this determinantal product state
        matrix element. """

        # Matrix element is zero if initial and final states differ in more electrons than the operator is acting on
        if self.num_diff > tensor_size:
            return

        # Calculate number of elementary elements
        size = math.comb(self.num_electrons - self.num_diff, tensor_size - self.num_diff)
        size *= math.factorial(tensor_size) ** 2

        # Store all tuples of initial and final electrons containing all combinations of tensor_size-num_diff
        # same electrons together with the fixed other electrons
        elements = np.zeros((size, 2 * tensor_size + 1), dtype=dtype)
        index = 0
        for initial_electrons, final_electrons in self.iterate(tensor_size):

            # Iterate through all permutations of the initial electrons with sign
            for initial, sign_initial in self.determinant(initial_electrons):

                # Iterate through all permutations of the final electrons with sign
                for final, sign_final in self.determinant(final_electrons):
                    # Store initial and final electrons with total sign
                    sign = (self.sign + sign_initial + sign_final) % 2
                    elements[index, :] = initial + final + (sign,)
                    index += 1

        # Return the array of elementary elements
        return elements


class ElementStorage:
    def __init__(self, index_dtype, element_dtype, index_size, element_cols):
        """ Create temporary HDF5 file with datasets 'indices' and 'elements'. """

        self.index_dtype = index_dtype
        self.element_dtype = element_dtype
        self.index_size = index_size
        self.element_cols = element_cols

        # Generate temporary HDF5 file
        fd, self.path = tempfile.mkstemp(suffix=".hdf5", prefix="elements_")
        os.close(fd)
        self.fp = h5py.File(self.path, 'w', libver='latest')

        # Initialise empty resizable dataset 'indices'
        self.indices = self.fp.create_dataset(
            "indices",
            shape=(self.index_size, 3),
            maxshape=(None, 3),
            dtype=self.index_dtype,
            chunks=(1024, 3),
            compression="gzip"
        )

        # Initialise empty resizable dataset 'elements'
        self.elements = self.fp.create_dataset(
            "elements",
            shape=(128 * 1024, self.element_cols),
            maxshape=(None, self.element_cols),
            dtype=self.element_dtype,
            chunks=(128 * 1024, self.element_cols),
            compression="gzip"
        )

        # Initialise index and element counters
        self.num_indices = 0
        self.num_elements = 0
        self.immutable = False

    def append(self, initial, final, elements):
        """ Append row to dataset 'indices' and chunk to dataset 'elements'. """

        assert not self.immutable
        assert len(elements.shape) == 2
        assert elements.shape[1] == self.element_cols

        size = int(elements.shape[0])
        self.indices[self.num_indices, :] = (initial, final, size)
        self.elements[self.num_elements:self.num_elements + size, :] = elements
        self.num_indices += 1
        self.num_elements += size

    def finalize(self):
        """ Fix final sizes of the datasets and make the object immutable. """

        assert not self.immutable
        self.indices.resize(self.num_indices, axis=0)
        self.elements.resize(self.num_elements, axis=0)
        self.immutable = True

    def close(self):
        """ Close and remove the temporary HDF5 file. """

        self.fp.close()
        if os.path.exists(self.path):
            os.remove(self.path)


class ProductElements:
    """ Support class for the calculation of matrix elements for m-electron tensor operators within a configuration
    of n electrons, where m <= n. Based on a given electron configuration the object generates a ProductState object
    for each state of the configuration. This allows for a fast determination of equal electrons for every pair of
    electrons. Since an m-electron matrix element is zero if the initial and final differ in more than m electrons,
    the information on equal electrons is used to identify all potentially non-zero matrix elements. A ProductState
    object is generated for each of those elements, which delivers the terms of the sum of matrix elements of
    elementary m-electron operators resembling the value of the respective matrix element of the high-level tensor
    operator.

    The method matrix_elements(m) provides the lists of high-level index pairs and elementary elements. Since
    these lists are physical constants and their calculation can be very time-consuming, they should be calculated
    only once and cached in a file. """

    def __init__(self, config):
        """ Initialize all ProductState objects. """

        # Electron configuration
        self.config = config
        self.num_states = self.config.num_states
        self.num_electrons = self.config.info.num_electrons

        # Prepare all product states of the configuration for the detection of matching electrons
        logger.debug(f"Initializing {self.num_states} ProductStates")
        self.states = [ProductState(state) for state in self.config.states.indices]
        logger.debug(f"Initialized all {self.num_states} ProductStates")

        # List of all potentially non-zero matrix elements
        self.unused = [(i, j) for i in range(self.num_states) for j in range(i, self.num_states)]
        self.elements = []
        self.len_diff = []

    def add_elements(self):
        """ Add all potentially non-zero elements for which initial and final state differ in num_diff electrons.
        For each call num_diff is effectively increased by 1, since the number of new elements is appended to
        the attribute self.len_diff."""

        # Get next number of different electrons
        num_diff = len(self.len_diff)
        logger.debug(f"Adding elements for {num_diff} different electrons ({len(self.unused)} tests)")

        # Determine all pairs of initial and final state which differ in num_diff electrons using and updating
        # the list of unused matrix element indices
        i = 0
        while i < len(self.unused):
            initial_index, final_index = self.unused[i]
            initial_state = self.states[initial_index]
            final_state = self.states[final_index]

            # Skip if this pair of initial and final state is no match for num_diff different electrons
            same_electrons = initial_state.same_electrons(num_diff) & final_state.same_electrons(num_diff)
            if len(same_electrons) > 1:
                RuntimeError("Code error: multiple matches!")
            if len(same_electrons) == 0:
                i += 1
                r = len(self.unused) - i
                if r % 10000 == 0:
                    n = len(self.elements)
                    logger.debug(f"  {self.config.name} | Number of elements: {n}, remaining tests: {r}")
                continue

            # Pick the tuple of matching (same) electrons from the intersection set
            same_electrons = same_electrons.pop()

            # Collect the non-matching (other) electrons from both states
            initial_other_electrons, initial_sign = initial_state.other_electrons(same_electrons)
            final_other_electrons, final_sign = final_state.other_electrons(same_electrons)

            # Combined sign of both states
            sign = (initial_sign + final_sign) % 2

            # Append a ProductElement object to the list of matrix elements
            product_element = ProductElement(same_electrons, initial_other_electrons, final_other_electrons, sign)
            self.elements.append((initial_index, final_index, product_element))
            self.unused.pop(i)

        # Append the number of potentially non-zero elements, which increments num_diff for the next function call
        self.len_diff.append(len(self.elements))
        if len(self.len_diff) < 2:
            sum = self.len_diff[0]
        else:
            sum = self.len_diff[-1] - self.len_diff[-2]
        logger.debug(f"Added all {sum} elements for {num_diff} different electrons")

    def matrix_elements(self, tensor_size):
        """ Generate and return the list (indices) of potentially non-zero matrix elements for a tensor operator
        acting on tensor_size electrons and a list of elementary matrix elements (elements) which must be evaluated
        to calculate the value of each of the matrix elements. """

        # Matrix element is zero if the initial and final state differ on more electrons as the operator is acting on
        if tensor_size > self.num_electrons:
            return [], []

        # Update the list of matching elements according to the given number of electrons, the elementary tensor is
        # acting on (tensor_size)
        while len(self.len_diff) <= tensor_size:
            self.add_elements()

        # Determine datatype and type code for the dataset 'indices' by its maximum element
        max_value = max(self.num_states, math.comb(self.num_electrons, tensor_size) * math.factorial(tensor_size) ** 2)
        index_dtype, match = get_dtype(max_value)
        assert match

        # Determine datatype and type code for the dataset 'elements' by its maximum element
        max_value = self.config.states.pool_size - 1
        element_dtype, match = get_dtype(max_value)
        assert match

        # Build the index list and the list of elementary matrix elements
        max_index = self.len_diff[tensor_size]
        element_cols = 2 * tensor_size + 1
        storage = ElementStorage(index_dtype, element_dtype, max_index, element_cols)
        logger.debug(f"Collecting {max_index} elements for {tensor_size}-electron operators")
        i = 0
        for initial_index, final_index, product_element in self.elements[:max_index]:
            elements = product_element.elementary(tensor_size, element_dtype)
            storage.append(initial_index, final_index, elements)

            # Log progress
            i += 1
            if i % 10000 == 0:
                logger.debug(f"  {self.config.name} | Collected {i}/{max_index} elements")

        # Return the list of matrix element indices and elementary tensor arguments
        logger.debug(f"Collected all {max_index} elements for {tensor_size}-electron operators")
        storage.finalize()
        return storage


###########################################################################
# Product class
###########################################################################

TITLE = "Support data for the calculation of product state matrix elements of spherical tensor operators"

DESCRIPTION = """
This container stores data to support the calculation of product state matrix elements for m-electron tensor
operators (m = attribute 'numTensorElectrons') within a configuration of n electrons (n = attribute
'config.numElectrons'), where m <= n.
Each matrix element is expressed as a sum of matrix elements from elementary tensor operators acting on m electrons.
The sign (-1)^s of each elementary matrix element in the sum is determined by its stored sign flag s: s = 0 indicates
a positive sign, while s = 1 indicates a negative sign.
<br> {states_desc}
<br>
Structure of the HDF5 container item 'data/product.hdf5':
Each row in the dataset 'indices' corresponds to a potentially non-zero matrix element of a high-level
m-electron tensor operator.
It consists of three elements: two state indices and the number of rows addressed in the dataset 'elements'.
The state indices are related to the list of single electron states of the configuration in 'states.hdf5'.
The dataset 'elements' is organized as a data stack.
Each row in the dataset 'indices' addresses a consecutive block of rows within 'elements'.
Each row in 'elements' represents one elementary tensor operator element mentioned above.
The row contains m indices for the initial electrons, m indices for the final electrons, and the sign flag of
the element.
The electron indices are linked to the single electron states in 'states.electronPool' in the item 'product.json'.
"""


class Product:
    """ Class providing support data for the calculation of product state matrix elements of tensor operators. """

    def __init__(self, config_name: str, tensor_size: int):
        """ Initialize the support data structure for tensor operators acting on tensor_size electrons. """

        # Configuration string
        self.config_name = config_name

        # Number of electrons the tensor is acting on
        self.tensor_size = tensor_size

        # Load data container
        self.file = self.get_path(config_name, tensor_size)
        if self.file not in container_vault:
            self.generate_container()
        dc = container_vault[self.file]

        # Extract UUID and code version from the container
        meta = dc["data/product.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Extract electron configuration from the container
        self.config = ConfigInfo.from_meta(meta["config"])
        assert self.config.name == config_name

        # Extract product states from the container
        self.states = space_registry["Product"].from_meta(dc["data/states.hdf5"], meta["states"])
        self.num_states = self.states.num_states

        # Support data from HDF5 data structure.
        # Note: Do not use decode_uint_array. It is too slow, memory consumption too high.
        self.indices = dc["data/product.hdf5"]["indices"]
        self.elements = dc["data/product.hdf5"]["elements"]
        self.num_indices = len(self.indices)
        self.num_elements = len(self.elements)

    def generate_container(self):
        """ Generate the product state support data for tensor operators acting on tensor_size electrons and store
        it in a data container file. """

        # Get electron configuration
        config = Config(self.config_name)
        assert 1 <= self.tensor_size <= config.info.num_electrons
        config_meta = config.info.as_meta()
        logger.info(f"Prepare {config.name} product states for {self.tensor_size} electrons")

        # Get product states
        states_dict, states_meta = config.states.as_meta()

        # Get ProductElements object
        product_elements = ProductElements(config)

        # Generate product state support data
        t = time.time()
        storage = product_elements.matrix_elements(self.tensor_size)
        # product_dict = {"indices": storage.indices, "elements": storage.elements}
        product_dict = {"indices": np.array(storage.indices), "elements": np.array(storage.elements)}

        # Prepare container description string
        kwargs = {
            "states": "states",
            "states_hdf5": "states.hdf5",
            "json": "product.json",
        }
        kwargs["states_desc"] = desc_format(config.states_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Container structure
        items = {
            "content.json": {
                "containerType": {"name": "ameliProduct"},
                "usedSoftware": [{"name": "AMELI", "version": "1.0.0",
                                  "id": "https://github.com/reincas/ameli", "idType": "URL"}],
            },
            "meta.json": {
                "title": TITLE,
                "description": description,
                "license": "cc-by-sa-4.0",
                "version": __version__,
            },
            "data/product.json": {
                "version": __version__,
                "config": config_meta,
                "states": states_meta,
                "numTensorElectrons": self.tensor_size,
            },
            "data/states.hdf5": states_dict,
            "data/product.hdf5": product_dict,
        }

        # Create the data container and store it in a file
        container_vault[self.file] = items

        # Delete the temporary HDF5 file
        storage.close()

        t = time.time() - t
        ts = self.tensor_size
        logger.info(f"Stored {config.name} product states for {ts} electrons ({t:.1f} seconds) -> {self.file}")

    def electrons(self, indices):
        """ Convert the given sequence of electron indices into a tuple of Electron objects. """

        return tuple(self.states.electron_pool[i] for i in indices)

    def matrix_elements(self):
        """ Generate indices of every potentially non-zero matrix element together with a generator function for the
        arguments for every elementary tensor matrix element. These arguments are the involved electrons and the
        sign of the elementary element. """

        # Elementary elements argument generator function
        def electron_generator(sub_elements):
            for args in sub_elements:
                electrons = self.electrons(args[:-1])
                sign = args[-1]
                yield electrons, sign

        # Generate indices of matrix elements together with the respective electron generator
        index = 0
        for initial_index, final_index, size in self.indices:
            initial_index = int(initial_index)
            final_index = int(final_index)
            size = int(size)
            yield initial_index, final_index, electron_generator(self.elements[index:index + size])
            index += size

    @staticmethod
    def get_path(config_name, tensor_size):
        """ Return data container file name. """

        return f"{config_name}/product_{tensor_size}.zdc"
