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

import re
from pathlib import Path

from ameli.config import ConfigContainer
from ameli.product import ProductContainer
from ameli.unit import UnitContainer
from ameli.matrix import MatrixContainer
from ameli.transform import TransformContainer
from ameli.vault import VAULT_PATH


def get_configs():
    """ Finds all lanthanide ion configuration folders 'f<num>' in the AMELI vault path and returns a list of the
    respective numbers of electrons. """

    nums = []
    pattern = re.compile(r'^f(\d+)$')

    for folder in VAULT_PATH.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                nums.append(int(match.group(1)))

    return sorted(nums)


def update_containers(num_electrons):
    """ Generate the classes and parameters necessary to update every data container for the given lanthanide
    configuration. """

    def matrix_generator(root_path, folder, space, reduced):
        """ Generate all matrix classes in the given folder. """

        folder = root_path / folder
        assert folder.is_dir()
        for file in folder.iterdir():
            assert file.suffix == ".zdc"
            head, *params = file.stem.split("_")
            if params:
                name = f"{head}/{','.join(params)}"
            else:
                name = head
            yield MatrixContainer, (config_name, name, space, reduced)

    # Prepare root folder of the given lanthanide configuration
    config_name = f"f{num_electrons}"
    root_path = VAULT_PATH / Path(config_name)

    # Yield Config class
    file = root_path / "config.zdc"
    assert file.exists()
    yield ConfigContainer, (config_name,)

    # Yield Product class
    for tensor_size in range(1, min(num_electrons, 3) + 1):
        file = root_path / f"product_{tensor_size}.zdc"
        assert file.exists()
        yield ProductContainer, (config_name, tensor_size)

    # Yield Unit class
    folder = root_path / "unit"
    assert folder.is_dir()
    for file in folder.iterdir():
        assert file.suffix == ".zdc"
        head, *params = file.stem.split("_")
        name = f"{head}/{','.join(params)}"
        yield UnitContainer, (config_name, name)

    # Yield Matrix class for product states
    yield from matrix_generator(root_path, "product", "Product", False)

    # Yield Transform class
    file = root_path / "transform.zdc"
    assert file.exists()
    yield TransformContainer, (config_name,)

    # Yield Matrix class for LS coupling
    yield from matrix_generator(root_path, "sljm", "SLJM", False)
    yield from matrix_generator(root_path, "slj", "SLJ", False)
    yield from matrix_generator(root_path, "slj_reduced", "SLJ", True)


def get_root_path(num_electrons):
    """ Return the root path object for the lanthanide ion with the given number of electrons. """

    config_name = f"f{num_electrons}"
    return VAULT_PATH / Path(config_name)


# Content mapping for each zip folder
ZIP_STRUCTURE = [
    ("product.zip", ["transform.zdc", "product/*.zdc"]),
    ("sljm.zip", ["sljm/*.zdc"]),
    ("slj.zip", ["slj/*.zdc", "slj_reduced/*.zdc"]),
    ("support.zip", ["config.zdc", "product*.zdc", "unit/*.zdc"]),
]


def get_zip_folders(root_path):
    """ Generate names and file lists for zip folders containing all data container files available for the lanthanide
    ion with the given root path. """

    # Generate list of data container files for each zip folder in the given order
    for zip_name, patterns in ZIP_STRUCTURE:
        files = []
        for pattern in patterns:
            for file_path in root_path.glob(pattern):
                files.append(file_path)
        yield zip_name, files


def get_matrix_heads(num_electrons):
    """ Return list of the header keys of all matrices available for the lanthanide ion with the given number of
    electrons. """

    # Prepare root folder of the given lanthanide configuration
    config_name = f"f{num_electrons}"
    root_path = VAULT_PATH / Path(config_name)

    # Generate list of data container files for each zip folder in the given order
    matrix_heads = set()
    for zip_name, patterns in ZIP_STRUCTURE:
        if zip_name == "support.zip":
            continue
        for pattern in patterns:
            for file_path in root_path.glob(pattern):
                if file_path.stem == "transform":
                    continue
                matrix_heads.add(file_path.stem.split("_", 1)[0])
    return matrix_heads
