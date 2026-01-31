##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the class Config, which# represents an electron
# configuration with one or more subshells and all of its product states.
#
# The module also provides the class ProductStates, which represents all
# product states of a configuration.
#
##########################################################################

import logging
import re
import time
from itertools import combinations, product
from collections import namedtuple

import sympy
import sympy as sp

from . import register_space, desc_format
from .uintarray import decode_uint_array, encode_uint_array
from .vault import container_vault

__version__ = "1.0.0"
logger = logging.getLogger("config")

# Spectral letters representing quantum numbers of an orbital angular momentum
SPECTRAL = "spdfghiklmnoqrtuvwxyz"

# Regular expression matching a single subshell in a configuration string
SUBSHELL = r"([@@@])([0-9]+) *".replace("@@@", SPECTRAL)

# Named tuple holding the parameters required to the electrons of a certain subshell in an electron pool
SubShell = namedtuple("SubShell", ["shell", "l", "num", "indexStart", "indexStop"])

# Representation of the quantum numbers of a single electron
Electron = namedtuple("Electron", ["shell", "l", "ml", "s", "ms"])


###########################################################################
# Product states class
###########################################################################

class ProductStates:
    """ This class represents a list of determinantal product states as indices into a pool of Electron objects. """

    # State space string
    state_space: str

    # Pool of single electron states
    electron_pool: list
    pool_size: int

    # List of states as sequences of indices referencing single electron states in the pool
    indices: list
    num_states: int

    @classmethod
    def from_meta(cls, states_dict, info_meta):
        """ Return a ProductStates object initialized from its data container dictionaries. """

        # Initialize empty ProductStates object
        states = cls.__new__(cls)

        # Extract state info
        states.state_space = info_meta["stateSpace"]
        pool = [{k: sp.S(v) for k, v in electron.items()} for electron in info_meta["electronPool"]]
        states.electron_pool = [Electron(**electron) for electron in pool]
        states.pool_size = info_meta["numPoolStates"]
        states.num_states = info_meta["numStates"]

        # Extract list of states
        states.indices = decode_uint_array(states_dict, "indices")

        # Sanity checks for redundant information
        assert states.state_space == "Product"
        assert states.state_space == states_dict["stateSpace"]
        assert states.pool_size == len(pool)
        assert states.num_states == len(states.indices)

        # Return ProductStates object
        return states

    def as_meta(self):
        """ Return the data container dictionaries representing this object. """

        # States dictionary
        states_dict = encode_uint_array(self.indices, "indices")
        states_dict["stateSpace"] = self.state_space

        # State info dictionary
        info_meta = {
            "stateSpace": self.state_space,
            "electronPool": [{k: str(v) for k, v in e._asdict().items()} for e in self.electron_pool],
            "numPoolStates": len(self.electron_pool),
            "numStates": self.num_states,
        }

        # Return dictionaries
        return states_dict, info_meta


###########################################################################
# Generation of the electron configuration data
###########################################################################

def generate_config(config_name):
    """ Generate and return all electron configuration data. """

    # Split the configuration string into subshells
    subshell_params = [(SPECTRAL.index(l), int(num)) for l, num in re.findall(SUBSHELL, config_name.lower())]
    subshells = []
    index = 0
    for shell, (l, num) in enumerate(subshell_params):
        subshell_size = 2 * (2 * l + 1)
        assert 0 < num < subshell_size
        subshells.append(SubShell(shell=shell, l=l, num=num, indexStart=index, indexStop=index + subshell_size))
        index += subshell_size
    logger.debug(f"  {config_name} | Subshells: {len(subshells)}")

    # Collect the pool of electron quantum numbers for all subshells
    electron_pool = []
    for subshell in subshells:
        shell = subshell.shell
        l = subshell.l
        s = sympy.Rational(1, 2)
        magnetic = reversed(list(product(range(-l, l + 1), (-s, +s))))
        quantum = [(sp.S(l), sp.S(ml), s, ms) for ml, ms in magnetic]
        electrons = [Electron(shell=shell, l=l, ml=ml, s=s, ms=ms) for l, ml, s, ms in quantum]
        assert len(electrons) == subshell.indexStop - subshell.indexStart
        electron_pool.extend(electrons)
    logger.debug(f"  {config_name} | Electron pool: {len(electron_pool)}")

    # Generate product states as all electron combinations of the configuration
    states = []
    for subshell in subshells:
        indices = range(subshell.indexStart, subshell.indexStop)
        states.append(combinations(indices, subshell.num))
    states = [sum(state, ()) for state in product(*states)]
    logger.debug(f"  {config_name} | Electron states: {len(states)}")

    # Return product states, the list of subshells, and the pool of electrons
    return states, subshells, electron_pool


###########################################################################
# ConfigInfo class
###########################################################################

class ConfigInfo:
    """ Data class for some configuration related meta data. """

    # Configuration string
    name = ""

    # List of SubShell objects representing all partly occupied subshells of the configuration
    subshells = []

    # Number of subshells
    num_subshells = 0

    # Total number of electrons in all subshells
    num_electrons = 0

    @classmethod
    def from_meta(cls, info_meta: dict):
        """ Initialize and return a ConfigInfo object from a data container dictionary. """

        # Initialize empty ConfigInfo object
        info = cls.__new__(cls)

        # Extract config metadata
        info.name = info_meta["name"]
        info.subshells = [SubShell(**subshell) for subshell in info_meta["subShells"]]
        info.num_subshells = info_meta["numSubShells"]
        info.num_electrons = info_meta["numElectrons"]

        # Sanity checks
        assert info.num_subshells == len(info.subshells)
        assert info.num_electrons == sum(subshell.num for subshell in info.subshells)

        # Return ConfigInfo object
        return info

    def as_meta(self) -> dict:
        """ Return the content of this object as data container dictionary. """

        return {
            "name": self.name,
            "subShells": [subshell._asdict() for subshell in self.subshells],
            "numSubShells": self.num_subshells,
            "numElectrons": self.num_electrons,
        }


###########################################################################
# Config class
###########################################################################

TITLE = "Many-electron configuration"

DESCRIPTION = """
This container stores data regarding a configuration of n electrons in one or more partly occupied subshells.
Each subshell is characterised by a quantum number l and a number of electrons.
It contributes 2(2l+1) electron states with distinct ml and ms quantum numbers to the pool of single-electron
states of the configuration in '{states}.electronPool'.
<br> {states_desc}
"""

# Description of the HDF5 container item holding the product states of a configuration
PRODUCT_DESC = """
The HDF5 item '{states_hdf5}' contains all electron product states.
Each state (row) within this array comprises an ordered sequence of indices that reference the pool of
single-electron states in '{states}.electronPool' in the JSON item '{json}'.
"""


class Config():
    """ Class of an electron configuration. Provides the quantum numbers for all product states. """

    # Description of the HDF5 container item holding the product states of a configuration required by the
    # transformation interface
    states_desc = PRODUCT_DESC

    # Product states are the reference, no transformation matrix
    matrix = None

    def __init__(self, config_name):
        """ Initialize the electron configuration. """

        # Configuration string
        self.name = config_name

        # Load or generate data container
        self.file = self.get_path(config_name)
        if self.file not in container_vault:
            self.generate_container()
        dc = container_vault[self.file]

        # Extract UUID and code version from the container
        meta = dc["data/config.json"]
        self.uuid = dc.uuid
        self.version = meta["version"]

        # Extract electron configuration from the container
        self.info = ConfigInfo.from_meta(dc["data/config.json"])
        assert self.name == self.info.name

        # Extract product states from the container
        self.states = self.states_from_meta(dc["data/states.hdf5"], meta["states"])
        self.num_states = self.states.num_states

    def generate_container(self):
        """ Generate an electron configuration and store it in a data container file. """

        logger.info(f"Generating {self.name} configuration")
        t = time.time()

        # Generate the configuration data
        states, subshells, electron_pool = generate_config(self.name)
        states_dict = encode_uint_array(states, "indices")
        states_dict["stateSpace"] = "Product"

        # Product states metadata
        states_meta = {
            "stateSpace": "Product",
            "electronPool": [{k: str(v) for k, v in e._asdict().items()} for e in electron_pool],
            "numPoolStates": len(electron_pool),
            "numStates": len(states),
        }

        # Prepare container description string
        kwargs = {
            "states": "states",
            "states_hdf5": "states.hdf5",
            "json": "config.json",
        }
        kwargs["states_desc"] = desc_format(self.states_desc, kwargs)
        description = desc_format(DESCRIPTION, kwargs)

        # Structure of data container
        items = {
            "content.json": {
                "containerType": {"name": "ameliConfiguration"},
                "usedSoftware": [{"name": "AMELI", "version": "1.0.0",
                                  "id": "https://github.com/reincas/ameli", "idType": "URL"}],
            },
            "meta.json": {
                "title": TITLE,
                "description": description,
                "license": "cc-by-sa-4.0",
                "version": __version__,
            },
            "data/config.json": {
                "version": __version__,
                "name": self.name,
                "subShells": [subshell._asdict() for subshell in subshells],
                "numSubShells": len(subshells),
                "numElectrons": sum(subshell.num for subshell in subshells),
                "states": states_meta,
            },
            "data/states.hdf5": states_dict,
        }

        # Create the data container and store it in a file
        container_vault[self.file] = items
        t = time.time() - t
        logger.info(f"Stored {self.name} configuration: ({t:.1f} seconds) -> {self.file}")

    def electrons(self, indices):
        """ Convert the given sequence of electron indices into a tuple of Electron objects. """

        return tuple(self.states.electron_pool[i] for i in indices)

    @staticmethod
    def states_from_meta(states_dict, info_meta):
        """ Return a ProductStates object initialized from its data container dictionaries. """

        return ProductStates.from_meta(states_dict, info_meta)

    def states_as_meta(self):
        """ Return the data container dictionaries representing the product states. """

        return self.states.as_meta()

    @staticmethod
    def get_path(config_name):
        """ Return data container file name. """

        return f"{config_name}/config.zdc"


# Register space of electron product states
register_space("Product", Config, lambda dtype, config_name: Config(config_name))
