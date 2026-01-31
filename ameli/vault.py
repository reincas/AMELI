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
import os
import time
from pathlib import Path

from scidatacontainer import Container

VAULT_PATH = Path(__file__).resolve().parent / "vault"


###########################################################################
# Vault class
###########################################################################

class Vault:
    """ Interface class for data container files. """

    def path(self, name: str) -> Path:
        """ Return Path object for the given file name. """

        return VAULT_PATH / Path(name)

    def __setitem__(self, name: str, items: dict):
        """ Create and store a SciDataContainer from the given items dictionary. """

        dc = Container(items=items)
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        dc.write(str(tmp_path))

        max_retries = 100
        for i in range(max_retries):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                time.sleep(0.05 * (i + 1))
        raise RuntimeError(f"Storage of {path} failed!")

    def __getitem__(self, name: str) -> Container:
        """ Load and return the SciDataContainer for the given file name. """

        path = self.path(name)
        return Container(file=str(path))

    def __contains__(self, name: str) -> bool:
        """ Returns True if the given file exists and False otherwise. """

        return self.path(name).exists()


# Global interface
container_vault = Vault()
