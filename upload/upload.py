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
RECORDS_SB = {
    1: {"concept_id": 489074, "version": "2.0.0"},
    2: {"concept_id": 489076, "version": "2.0.0"},
    3: {"concept_id": 489078, "version": "2.0.0"},
    4: {"concept_id": 489080, "version": "2.0.0"},
    5: {"concept_id": 489082, "version": "2.0.0"},
    6: {"concept_id": 489084, "version": "2.0.0"},
    7: {"concept_id": 489086, "version": "2.0.0"},
    8: {"concept_id": 489088, "version": "2.0.0"},
    9: {"concept_id": 489090, "version": "2.0.0"},
    10: {"concept_id": 489092, "version": "2.0.0"},
    11: {"concept_id": 489094, "version": "2.0.0"},
    12: {"concept_id": 489096, "version": "2.0.0"},
    13: {"concept_id": 489098, "version": "2.0.0"},
}
RECORDS = {
    1: {"concept_id": 19130697, "version": "2.0.0"},
    2: {"concept_id": 19139480, "version": "2.0.0"},
    3: {"concept_id": 19144764, "version": "2.0.0"},
    4: {"concept_id": 19154321, "version": "2.0.0"},
    5: {"concept_id": 19154326, "version": "2.0.0"},
    6: {"concept_id": 19158643, "version": "2.0.0"},
    7: {"concept_id": 19158647, "version": "2.0.0"},
    8: {"concept_id": 19158658, "version": "2.0.0"},
    9: {"concept_id": 19158660, "version": "2.0.0"},
    10: {"concept_id": 19158667, "version": "2.0.0"},
    11: {"concept_id": 19158671, "version": "2.0.0"},
    12: {"concept_id": 19158675, "version": "2.0.0"},
    13: {"concept_id": 19158677, "version": "2.0.0"},
}

if __name__ == "__main__":

    for num_electrons in range(1, 14):
        if SANDBOX:
            version = RECORDS_SB[num_electrons]["version"]
            concept_id = RECORDS_SB[num_electrons]["concept_id"]
        else:
            version = RECORDS[num_electrons]["version"]
            concept_id = RECORDS[num_electrons]["concept_id"]
        zenodo_lanthanide(num_electrons, version, concept_id, SANDBOX)
