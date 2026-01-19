##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the class Vault, which represents a file
# cache for SciDataContainer objects.
#
##########################################################################

from functools import lru_cache
from pathlib import Path

from scidatacontainer import Container

VAULT_PATH = Path(__file__).resolve().parent / "vault"


###########################################################################
# Vault class
###########################################################################

class Vault:
    """ File cache for SciDataContainer files. """

    def __init__(self, config_name: str):
        """ Store electron configuration name. """

        self.config_name = Path(config_name)

    def path(self, name: str) -> Path:
        """ Return Path object for the given file name. """

        return VAULT_PATH / self.config_name / Path(name)

    def __setitem__(self, name: str, items: dict):
        """ Create and store a SciDataContainer from the given items dictionary. """

        dc = Container(items=items)
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        dc.write(path)

    def __getitem__(self, name: str) -> Container:
        """ Load and return the SciDataContainer for the given file name. """

        path = self.path(name)
        return Container(file=path)

    def __contains__(self, name: str) -> bool:
        """ Returns True if a SciDataContainer for given file name is present and False otherwise. """

        path = VAULT_PATH / self.config_name / name
        return path.exists()


@lru_cache(maxsize=1)
def get_vault(config_name):
    """ Return cached Vault object. """

    return Vault(config_name)
