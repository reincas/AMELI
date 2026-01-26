##########################################################################
# Copyright (c) 2025-2026 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import re


##########################################################################
# Transformation space registry
##########################################################################

class StateSpace:
    """ Interface class for the abstraction of different coupling spaces. """

    def __init__(self, name, cls, get_transform, subspace=None):
        """ Store state space name, transformation class and transformation object getter function. """

        self.name = name
        self.cls = cls
        self.get_transform = get_transform
        self.subspace = subspace

        # No transformation object loaded yet
        self.transform = None

        # Sanity check for the transformation object interface
        attributes = ("states_desc", "states_from_meta", "states_as_meta", "matrix")
        for attr in attributes:
            assert hasattr(self.cls, attr), f"Attribute {self.cls.__name__}.{attr} missing!"

    def load(self, dtype, config_name):
        """ Load a transformation object. """

        self.transform = self.get_transform(dtype, config_name)

    @property
    def states_desc(self):
        """ Return meta data description string for the states. """

        if isinstance(self.cls.states_desc, dict):
            subspace = self.subspace or self.name
            return self.cls.states_desc[subspace]

        assert self.subspace is None, "Subspace description is missing!"
        return self.cls.states_desc

    def from_meta(self, states_dict, info_meta):
        """ Return a state object from the given meta data dictionaries. """

        return self.cls.states_from_meta(states_dict, info_meta)

    @property
    def matrix(self):
        """ Return the transformation matrix from the product state space to this coupling. This function requires
        a loaded transformation object. """

        assert self.transform is not None, "Load transform object first!"
        return self.transform.matrix

    def as_meta(self):
        """ Return the meta data dictionaries of the state space. This function requires a loaded transformation
        object. """

        assert self.transform is not None, "Load transform object first!"
        return self.transform.states_as_meta()


# Initialise empty state space registry
space_registry = {}


def register_space(space, cls, get_transform):
    """ Register the state space name, transformation class and transformation object getter function for the given
    state space. """

    space_registry[space] = StateSpace(space, cls, get_transform)


def register_subspace(space, subspace):
    """ Register a subspace to the given state space. """

    assert space in space_registry, f"State space {space} is not registered!"
    parent = space_registry[space]
    space_registry[subspace] = StateSpace(space, parent.cls, parent.get_transform, subspace=subspace)


##########################################################################
# Description string formatting
##########################################################################

def desc_format(description, kwargs):
    """ Basic transformation for mata data description strings. """

    desc = str(description)
    desc = re.sub(r"\s*\n\s*", " ", desc)
    desc = re.sub(r"\s*<br>\s*", "\n", desc)
    desc = desc.strip()
    desc = desc.format(**kwargs)
    return desc


def lanthanide_matrices():
    """ Return a list of names of all available matrices for lanthanide ions and the number of electron each is
    acting on. """

    names = []
    names.extend([(f"U/{k},{q}", 1) for k in range(7) for q in range(-k, k + 1)])
    names.extend([(f"T/{k},{q}", 1) for k in range(2) for q in range(-k, k + 1)])
    names.extend([(f"UU/{k}", 1) for k in (0, 1, 2, 3, 4, 5, 6)])
    names.extend([(f"TT/{k}", 1) for k in (0, 1)])
    names.extend([(f"UT/{k}", 1) for k in (0, 1)])
    names.extend([(f"L/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"S/{q}", 1) for q in range(-1, 2)])
    names.extend([(f"J/{q}", 1) for q in range(-1, 2)])
    names.append(("L2", 1))
    names.append(("S2", 1))
    names.append(("J2", 1))
    names.append(("LS", 1))
    names.extend([(f"H1/{k}", 2) for k in (2, 4, 6)])
    names.append(("H2", 1))
    names.extend([(f"H3/{i}", 2) for i in (0, 1, 2)])
    names.extend([(f"H4/{c}", 3) for c in (2, 3, 4, 6, 7, 8)])
    names.extend([(f"Hss/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"Hsoo/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H5/{k}", 2) for k in (0, 2, 4)])
    names.extend([(f"H6/{k}", 2) for k in (0, 2, 4, 6)])
    return names


##########################################################################
# Import main classes
##########################################################################

from .config import SPECTRAL, Electron, Config, get_config
from .product import Product, get_product
from .unit import Unit
from .matrix import Matrix
from .transform import Transform, get_transform
