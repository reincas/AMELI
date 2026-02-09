##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the state space registry.
#
##########################################################################

class StateSpace:
    """ Interface class for the abstraction of different coupling spaces. """

    def __init__(self, name, cls, subspace=None):
        """ Store state space name, transformation class and transformation object getter function. """

        self.name = name
        self.cls = cls
        self.subspace = subspace

        # No transformation object loaded yet
        self.transform = None

        # Sanity check for the transformation object interface
        attributes = ("states_desc", "states_from_meta", "states_as_meta", "matrix")
        for attr in attributes:
            assert hasattr(self.cls, attr), f"Attribute {self.cls.__name__}.{attr} missing!"

    def load(self, config_name):
        """ Load a transformation object. """

        self.transform = self.cls(config_name)

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

    def as_meta(self, hasher):
        """ Return the meta data dictionaries of the state space. This function requires a loaded transformation
        object. """

        assert self.transform is not None, "Load transform object first!"
        return self.transform.states_as_meta(hasher)


# Initialise empty state space registry
space_registry = {}


def register_space(space, cls):
    """ Register the state space name, transformation class and transformation object getter function for the given
    state space. """

    space_registry[space] = StateSpace(space, cls)


def register_subspace(space, subspace):
    """ Register a subspace to the given state space. """

    assert space in space_registry, f"State space {space} is not registered!"
    parent = space_registry[space]
    space_registry[subspace] = StateSpace(space, parent.cls, subspace=subspace)


