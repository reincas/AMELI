##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
from ameli.lanthanide.content import get_configs, update_containers

for num_electrons in get_configs():
    for cls, params in update_containers(num_electrons):
        print(f"{cls.__name__}({", ".join(map(str, params))})")
        obj = cls(*params)
