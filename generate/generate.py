##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import time
import sys

from build import run_sync, run_pool

POOL = True


##########################################################################
# Catch process kill
##########################################################################

def handle_exception(exc_type, exc_value, exc_traceback):
    """ Safety net for any crash that Python doesn't catch. """

    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


##########################################################################
# Main program
##########################################################################

if __name__ == "__main__":
    t = time.time()

    if len(sys.argv) > 1:
        nums = [int(sys.argv[1])]
        file = f"ameli-{nums[0]}.log"
    else:
        nums = [1, 13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]
        file = f"ameli.log"

    # Log format
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File Handler
    file_h = logging.FileHandler(file, mode="a")
    file_h.setFormatter(log_format)
    file_h.setLevel(logging.DEBUG)

    # Console Handler
    console_h = logging.StreamHandler()
    console_h.setFormatter(log_format)
    console_h.setLevel(logging.DEBUG)

    handlers = (file_h, console_h)
    if POOL:
        run_pool(nums, handlers)
    else:
        run_sync(nums, handlers)
