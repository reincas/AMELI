##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from ameli.vault import VAULT_PATH
from ameli.lanthanide.zenodo import ZenodoBucket

if __name__ == "__main__":

    num_electrons = 2
    config_name = f"f{num_electrons}"

    title = f"Test Record {config_name}"
    desc = "My description follows..."
    zenodo = ZenodoBucket(title, desc, sandbox=True)
    path = VAULT_PATH / config_name / "config.zdc"
    zenodo.upload_file(path)
    print(f"Zenodo URL: {zenodo.zenodo_url}")
