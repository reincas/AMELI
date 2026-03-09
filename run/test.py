##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import configparser
import datetime
from pathlib import Path
import pytest
import sys

CODES = {
    0: "All tests passed",
    1: "Some tests failed",
    2: "Interrupted (Ctrl+C)",
    3: "Internal error",
    4: "Usage error",
    5: "No tests collected",
}


def main(test_args=None):
    test_path = Path("../test")
    path = test_path / "pytest.ini"

    config = configparser.ConfigParser()
    config.read(str(path))
    log_file = config.get('pytest', 'log_file')

    start_time = datetime.datetime.now()
    print(f"{start_time} --- Starting AMELI Test Suite (log: {log_file}) ---")

    if test_args:
        pytest_target = str(test_path / test_args["file"])
        if "func" in test_args:
            pytest_target += f'::{test_args["func"]}'
            if "args" in test_args:
                # Pytest uses brackets for parametrized values
                pytest_target += f'[{test_args["args"]}]'
    else:
        pytest_target = str(test_path)

    args = [pytest_target, "-v", "--durations=0"]
    exit_code = pytest.main(args)

    if exit_code in CODES:
        exit_msg = f"{exit_code} ({CODES[exit_code]})"
    else:
        exit_msg = f"{exit_code}"
    end_time = datetime.datetime.now()
    print(f"{end_time} --- AMELI Test Suite Finished with Exit Code {exit_msg} ---")
    sys.exit(exit_code)


if __name__ == "__main__":
    test_args = None
    #test_args = {"file": "test_03_states.py", "func": "test_states", "args": 5}
    main(test_args)
