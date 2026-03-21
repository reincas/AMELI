##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Note: This script is for documentation only. It is required by the
# AMELI maintainer to upload the results of the AMELI package to the
# Zenodo repository.
#
##########################################################################

from zenodo import zenodo_lanthanide

SANDBOX = False
RECORDS = {
    1: {"concept_id": 19130697, "version": "1.0.0"},
    2: {"concept_id": 19139480, "version": "1.0.0"},
    3: {"concept_id": None, "version": "1.0.0"},
    4: {"concept_id": None, "version": "1.0.0"},
    5: {"concept_id": None, "version": "1.0.0"},
    6: {"concept_id": None, "version": "1.0.0"},
    7: {"concept_id": None, "version": "1.0.0"},
    8: {"concept_id": None, "version": "1.0.0"},
    9: {"concept_id": None, "version": "1.0.0"},
    10: {"concept_id": None, "version": "1.0.0"},
    11: {"concept_id": None, "version": "1.0.0"},
    12: {"concept_id": None, "version": "1.0.0"},
    13: {"concept_id": None, "version": "1.0.0"},
}

if __name__ == "__main__":

    for num_electrons in [1, 2]:
        version = RECORDS[num_electrons]["version"]
        concept_id = RECORDS[num_electrons]["concept_id"]
        zenodo_lanthanide(num_electrons, version, concept_id, SANDBOX)