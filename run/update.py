##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This script loads every data container in the vault to trigger a
# potential container update.
#
##########################################################################

from datetime import datetime

from ameli.lanthanide.content import get_configs, update_containers

for num_electrons in get_configs():
    for cls, params in update_containers(num_electrons):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} - {cls.__name__}({", ".join(map(str, params))})")
        obj = cls(*params)
