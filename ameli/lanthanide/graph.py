##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the generation of a directed cyclic graph
# representing the dependencies of data container objects (nodes).
#
##########################################################################

import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import sympy as sp

from ameli.config import generate_config, ConfigInfo, Config
from ameli.product import Product
from ameli.unit import MATRIX, Unit
from ameli.matrix import FORCE_SYMBOLIC, MatrixName, Matrix
from ameli.transform import SYM_INFO, SYM_CHAIN, config_key, Transform
from ameli.datatype import DataType
from ameli.vault import container_vault


##########################################################################
# Mocked Config class for the Node objects
##########################################################################

class MockConfig:
    def __init__(self, name):
        self.name = name

        # Generate configuration data
        states, subshells, electron_pool = generate_config(self.name)
        # states_dict = encode_uint_array(states, "indices")

        # Build info stricture
        info_meta = {
            "name": name,
            "subShells": [subshell._asdict() for subshell in subshells],
            "numSubShells": len(subshells),
            "numElectrons": sum(subshell.num for subshell in subshells),
        }
        self.info = ConfigInfo.from_meta(info_meta)

        # Build states structure
        states_dict = {"indices": np.array(states), "stateSpace": "Product"}
        states_meta = {
            "stateSpace": "Product",
            "electronPool": [{k: str(v) for k, v in e._asdict().items()} for e in electron_pool],
            "numPoolStates": len(electron_pool),
            "numStates": len(states),
        }
        self.states = Config.states_from_meta(states_dict, states_meta)
        self.num_states = self.states.num_states


##########################################################################
# Dry run function to detect matrix dependencies
##########################################################################

def matrix_dry_run(func, *args, **kwargs):
    """ Run given matrix creation function with mocked classes Unit and Matrix. Return initialisation arguments of
    every created Unit and Matrix object. """

    # Initialise list of arguments and dummy config object
    trace = []
    config = args[1]

    def mock_unit(dtype, config_name, name):
        """ Return stunt double of a Matrix object. """

        # Store Uatrix initialisation arguments
        kwargs = {
            "dtype": dtype.name,
            "config_name": config_name,
            "name": name,
        }
        trace.append(("Unit", kwargs))

        # Return stunt double for the Unit object with empty matrix
        stunt_double = MagicMock()
        num_states = config.num_states
        stunt_double.matrix = sp.zeros(num_states, num_states)
        return stunt_double

    def mock_matrix(dtype, config_name, name, state_space, reduced=False):
        """ Return stunt double of a Matrix object. """

        # Store Matrix initialisation arguments
        kwargs = {
            "dtype": dtype.name,
            "config_name": config_name,
            "name": name,
            "state_space": state_space,
            "reduced": reduced,
        }
        trace.append(("Matrix", kwargs))

        # Return stunt double for the Matrix object with empty matrix
        stunt_double = MagicMock()
        num_states = config.num_states
        stunt_double.matrix = sp.zeros(num_states, num_states)
        return stunt_double

    # Run given matrix creation function with mocked classes Unit and Matrix
    with patch("ameli.matrix.Unit", side_effect=mock_unit), \
            patch("ameli.matrix.Matrix", side_effect=mock_matrix):
        func(*args, **kwargs)

    # Return list of all initialised Unit and Matrix objects
    return trace


##########################################################################
# Container node registry
##########################################################################

class Registry:
    """ Dictionary of container nodes (Config, Product, Unit, Matrix, and Transform). Keys are hash strings based on
    class name and initialization arguments. """

    def __init__(self):
        """ Initialise nodes dictionary."""

        self.nodes = {}

    @staticmethod
    def get_id(cls, **kwargs):
        """ Return a unique, stable string ID for a node. """

        payload = {"cls": cls.__name__, "args": kwargs}
        dump = json.dumps(payload, sort_keys=True)
        return hashlib.md5(dump.encode()).hexdigest()

    def register(self, cls, **kwargs):
        """ Register and return the given node. """

        node_id = self.get_id(cls, **kwargs)
        if node_id not in self.nodes:
            self.nodes[node_id] = cls(self, node_id, **kwargs)
        return self.nodes[node_id]

    def weight(self, node_id):
        """ Return weight of the given node. """

        return self.nodes[node_id].weight

    def exists(self, node_id):
        """ Return True if the given node file exists. """

        return self.nodes[node_id].exists

    @property
    def unfinished(self):
        """ Return number of unfinished nodes. """

        return len([nid for nid, node in self.nodes.items() if not node.exists])

##########################################################################
# Container node classes
##########################################################################

class Node:
    def __init__(self, registry, node_id, cls, file, **kwargs):
        self.kwargs = kwargs
        self.registry = registry
        self.node_id = node_id
        self.ameli_cls = cls
        self.file = Path(file)
        self.parents = []
        self.children = []

        signature = inspect.signature(cls.__init__)
        self.arg_names = list(signature.parameters.keys())[1:]
        self.exists =  str(self.file) in container_vault

    @property
    def in_degree(self):
        return len([node_id for node_id in self.parents if not self.registry.nodes[node_id].exists])

    @property
    def out_degree(self):
        return len([node_id for node_id in self.children if not self.registry.nodes[node_id].exists])

    def __str__(self):
        name = self.ameli_cls.__name__
        args = [self.kwargs[name] for name in self.arg_names]
        args = [f'"{value}"' if isinstance(value, str) else value for value in args]
        args = ", ".join(f"{value}" for value in args)
        return f"{name}({args})"

    def register_parent(self, cls, **kwargs):
        parent = self.registry.register(cls, **kwargs)
        parent.register_child(self)
        self.parents.append(parent.node_id)

    def register_child(self, child):
        assert child.node_id is not None
        self.children.append(child.node_id)

    def generate(self):
        return self.ameli_cls(**self.kwargs)

    @property
    def weight(self):
        return 1 + sum(self.registry.weight(child_id) for child_id in self.children)

class ConfigNode(Node):
    def __init__(self, registry, node_id, config_name):
        kwargs = {
            "config_name": config_name,
        }
        file = Config.get_path(config_name)
        super().__init__(registry, node_id, Config, file, **kwargs)


class ProductNode(Node):
    def __init__(self, registry, node_id, config_name, tensor_size):
        kwargs = {
            "config_name": config_name,
            "tensor_size": tensor_size,
        }
        file = Product.get_path(config_name, tensor_size)
        super().__init__(registry, node_id, Product, file, **kwargs)

        self.register_parent(ConfigNode, config_name=config_name)


class UnitNode(Node):
    def __init__(self, registry, node_id, dtype, config_name, name):
        if isinstance(dtype, str):
            dtype = DataType(dtype)
        kwargs = {
            "dtype": dtype.name,
            "config_name": config_name,
            "name": name,
        }
        file = Unit.get_path(dtype, config_name, name)
        super().__init__(registry, node_id, Unit, file, **kwargs)

        self.register_parent(ConfigNode, config_name=config_name)
        key, parameters = name.split("/")
        _, tensor_size = MATRIX[key]
        self.register_parent(ProductNode, config_name=config_name, tensor_size=tensor_size)


class TransformNode(Node):
    def __init__(self, registry, node_id, dtype, config_name):

        if isinstance(dtype, str):
            dtype = DataType(dtype)
        kwargs = {
            "dtype": dtype.name,
            "config_name": config_name,
        }
        file = Transform.get_path(dtype, config_name)
        super().__init__(registry, node_id, Transform, file, **kwargs)

        self.register_parent(ConfigNode, config_name=config_name)

        config = MockConfig(config_name)
        chain = SYM_CHAIN[config_key(config)]["chain"]
        names = [SYM_INFO[name]["matrix"] for name in chain if SYM_INFO[name]["matrix"] is not None]
        for name in names:
            kwargs = {
                "dtype": dtype.name,
                "config_name": config_name,
                "name": name,
                "state_space": "Product",
                "reduced": False,
            }
            self.register_parent(MatrixNode, **kwargs)


class MatrixNode(Node):
    def __init__(self, registry, node_id, dtype, config_name, name, state_space, reduced=False):
        if reduced:
            assert state_space == "SLJ"
        if isinstance(dtype, str):
            dtype = DataType(dtype)
        kwargs = {
            "dtype": dtype.name,
            "config_name": config_name,
            "name": name,
            "state_space": state_space,
            "reduced": reduced,
        }
        file = Matrix.get_path(dtype, config_name, name, state_space, reduced)
        super().__init__(registry, node_id, Matrix, file, **kwargs)

        self.register_parent(ConfigNode, config_name=config_name)
        if FORCE_SYMBOLIC and not dtype.is_symbolic:
            kwargs = {
                "dtype": "symbolic",
                "config_name": config_name,
                "name": name,
                "state_space": state_space,
                "reduced": reduced,
            }
            self.register_parent(MatrixNode, **kwargs)
        elif state_space == "SLJ" and not reduced:
            kwargs = {
                "dtype": dtype.name,
                "config_name": config_name,
                "name": name,
                "state_space": "SLJM",
                "reduced": False,
            }
            self.register_parent(MatrixNode, **kwargs)
        elif state_space == "SLJ" and reduced:
            name_data = MatrixName(name)
            for name in name_data.components():
                kwargs = {
                    "dtype": dtype.name,
                    "config_name": config_name,
                    "name": name,
                    "state_space": "SLJ",
                    "reduced": False,
                }
                self.register_parent(MatrixNode, **kwargs)
        else:
            config = MockConfig(config_name)
            name_data = MatrixName(name)
            for mtype, kwargs in matrix_dry_run(name_data.func, dtype, config, *name_data.args):
                node = {"Unit": UnitNode, "Matrix": MatrixNode}[mtype]
                self.register_parent(node, **kwargs)
            if state_space == "SLJM":
                self.register_parent(TransformNode, dtype=dtype.name, config_name=config_name)
