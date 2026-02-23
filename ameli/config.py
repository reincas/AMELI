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

import hashlib
import logging
import re
import time
from itertools import combinations, product
from collections import namedtuple
import sympy as sp

from . import desc_format
from .states import register_space
from .uintarray import decode_uint_array, encode_uint_array
from .vault import AMELI_VERSION, VersionError, Vault

__version__ = "1.0.1"
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

    def __init__(self, electron_pool: list, indices: list):

        # State space string
        self.state_space = "Product"

        # Pool of single electron states
        self.electron_pool = electron_pool
        self.pool_size = len(electron_pool)

        # List of states as sequences of indices referencing single electron states in the pool
        self.indices = indices
        self.num_states = len(indices)

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

    @staticmethod
    def hash_data(hasher, states_dict, info_meta):
        """ Update hasher with representative state data. """

        # Update hasher with states_dict
        for key in sorted(states_dict.keys()):
            value = states_dict[key]
            hasher.update(key.encode('utf-8'))
            if key == "stateSpace":
                hasher.update(value.encode('utf-8'))
            else:
                hasher.update(value.tobytes())

        # Update hasher with info_meta
        for key in sorted(info_meta.keys()):
            value = info_meta[key]
            hasher.update(key.encode('utf-8'))
            if key == "electronPool":
                for electron in value:
                    for k in sorted(electron.keys()):
                        hasher.update(k.encode('utf-8'))
                        hasher.update(electron[k].encode('utf-8'))
            else:
                hasher.update(str(value).encode('utf-8'))


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
        s = sp.Rational(1, 2)
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
This container contains data regarding the electron configuration {config_name}.
The attribute '{states}.electronPool' in the JSON item {json} contains the quantum numbers of all available 
single-electron states.
<br> {states_desc}
"""

# Description of the HDF5 container item holding the product states of a configuration
PRODUCT_DESC = """
The HDF5 item '{states_hdf5}' contains all many-electron product states.
Each state (row) within this array comprises an ordered sequence of indices that reference the pool of
single-electron states in '{states}.electronPool' in the JSON item '{json}'.
"""


class ConfigContainer(Vault):
    """ Class representing a config data container. """

    # Description of the HDF5 container item holding the product states of a configuration required by the
    # transformation interface
    states_desc = PRODUCT_DESC

    def __init__(self, config_name):
        """ Provide the data container. """

        # Configuration string
        self.name = config_name

        # Load, update, or generate data container
        self.file = self.get_path(config_name)
        self.update_container(self.file, __version__)

    def generate_container(self, dc=None):
        """ Generate an electron configuration and store it in a data container file. Update function: Use product
        states from the given data container if provided. """

        if dc:
            v = dc["meta.json"]["version"]
            logger.info(f"Update {self.name} configuration (version {v} -> {__version__})")
        else:
            logger.info(f"Generate {self.name} configuration (version {__version__})")
        t = time.time()

        # Initialize the data hasher
        hasher = hashlib.sha256()

        # Generate the configuration data
        indices, subshells, electron_pool = generate_config(self.name)

        # Get product states from data container or determine them from scratch
        if dc:
            states = ProductStates.from_meta(dc["data/states.hdf5"], dc["data/config.json"]["states"])
        else:
            states = ProductStates(electron_pool, indices)
        states_dict, states_meta = states.as_meta()
        states.hash_data(hasher, states_dict, states_meta)

        # Generate data hash
        data_hash = hasher.hexdigest()
        if dc and "sha256Data" in dc["content.json"] and data_hash != dc["content.json"]["sha256Data"]:
            raise VersionError

        # Prepare container description string
        kwargs = {
            "config_name": self.name,
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
        self.write_container(self.file, items)
        t = time.time() - t
        logger.info(f"Stored {self.name} configuration: ({t:.1f} seconds) -> {self.file}")

    @staticmethod
    def get_path(config_name):
        """ Return data container file name. """

        return f"{config_name}/config.zdc"


class Config(ConfigContainer):
    """ Class of an electron configuration. Provides the quantum numbers for all product states. """

    # Transformation matrix
    matrix = None

    def __init__(self, config_name):
        """ Initialize the electron configuration. """

        # Load, update, or generate data container
        super().__init__(config_name)
        dc = self.read_container(self.file)

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

        # Product states are the reference, no transformation matrix
        self.matrix = None

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
    def hash_data(hasher, states_dict, info_meta):
        """ Update hasher with representative state data. """

        ProductStates.hash_data(hasher, states_dict, info_meta)


# Register space of electron product states
register_space("Product", Config)
