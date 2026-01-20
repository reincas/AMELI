##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging


##########################################################################
# Log handlers
##########################################################################

def log_console(formatter, level):
    filename = "stdout"
    logger = logging.getLogger()
    handler = [handler for handler in logger.handlers if handler.name == filename]
    if handler:
        handler = handler[0]
    else:
        handler = logging.StreamHandler()
        handler.set_name = filename
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    handler.setLevel(level)


def log_file(filename, formatter, level, delete=False):
    logger = logging.getLogger()
    handler = [handler for handler in logger.handlers if handler.name == filename]
    if handler:
        handler = handler[0]
    else:
        mode = "w" if delete else "a"
        handler = logging.FileHandler(filename, mode=mode)
        handler.set_name = filename
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    handler.setLevel(level)


