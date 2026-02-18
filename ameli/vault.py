##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This internal module provides the class Vault, which represents a file
# cache for SciDataContainer objects.
#
# This module also provides the class Version representing a semantic
# version number with major, minor, and patch level.
#
##########################################################################
import json
import os
import re
import time
import zipfile
from pathlib import Path

from scidatacontainer import Container

AMELI_VERSION = "1.2.0"
VERSION_PATTERN = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"
VAULT_PATH = Path(__file__).resolve().parent / "vault"


class VersionError(Exception):
    """ Raised when there is a version mismatch. """
    pass


class Version:
    """ Data class representing a valid semantic version number. """

    def __init__(self, version_str: str):
        match = re.match(VERSION_PATTERN, version_str.strip())

        # Sanity check
        if not match:
            raise ValueError(f"'{version_str}' is not a valid semantic version (major.minor.patch)")

        self.major = int(match.group(1))
        self.minor = int(match.group(2))
        self.patch = int(match.group(3))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f'Version("{self}")'

    @property
    def is_initial(self) -> bool:
        return str(self) == "1.0.0"

    @property
    def is_release(self) -> bool:
        return self.patch == 0

    @property
    def release(self) -> "Version":
        if self.is_release:
            return self
        return Version(f"{self.major}.{self.minor}.0")

    def same_release(self, other: "Version | str") -> bool:
        if isinstance(other, str):
            other = Version(other)
        if not isinstance(other, Version):
            return NotImplemented
        return self.release == other.release

    def next_version(self) -> "Version":
        return Version(f"{self.major}.{self.minor}.{self.patch + 1}")

    def next_release(self):
        return Version(f"{self.major}.{self.minor + 1}.0")

    def next_major(self):
        return Version(f"{self.major + 1}.0.0")

    @property
    def _tuple(self) -> tuple:
        return (self.major, self.minor, self.patch)

    def __eq__(self, other: "Version | str") -> bool:
        if isinstance(other, str):
            other = Version(other)
        if not isinstance(other, Version):
            return NotImplemented
        return self._tuple == other._tuple

    def __lt__(self, other: "Version | str") -> bool:
        if isinstance(other, str):
            other = Version(other)
        if not isinstance(other, Version):
            return NotImplemented
        return self._tuple < other._tuple

    def __gt__(self, other: "Version | str") -> bool:
        if isinstance(other, str):
            other = Version(other)
        if not isinstance(other, Version):
            return NotImplemented
        return self._tuple > other._tuple


###########################################################################
# Vault class
###########################################################################

class Vault:
    """ Interface class for data container files. """

    def write_container(self, name: str, items: dict):
        """ Store given items as data container file. """

        # Generate the data container object with hash
        dc = Container(items=items)
        dc.freeze()

        # Store data container in a temporary file
        path = self.vault_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        dc.write(str(tmp_path))

        # Wait until the final container file is completely stored and visible on the file system.
        # Note: os.replace() is an atomic operation on Windows and Linux, it never leaves an unfinished file.
        max_retries = 50
        for i in range(max_retries):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(min(1.0, 0.01 * (2 ** i)))
            except OSError as exp:
                raise RuntimeError(f"Disk I/O error: {exp}")

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Storage of {path} failed!")

    @staticmethod
    def read_container(name: str) -> Container:

        path = Vault.vault_path(name)
        return Container(file=str(path))

    @staticmethod
    def in_vault(name: str) -> bool:
        return Vault.vault_path(name).exists()

    @staticmethod
    def vault_path(name: str) -> Path:
        """ Return Path object for the given file name. """

        return VAULT_PATH / Path(name)

    def generate_container(self, dc=None):
        """ Implemented in sub-classes. """

        raise NotImplementedError()

    @staticmethod
    def container_version(name: str) -> Version:
        """ Extract and return the code version of the given data container without loading it completely. """

        with zipfile.ZipFile(Vault.vault_path(name), 'r') as z:
            with z.open('meta.json') as f:
                data = json.load(f)
                return Version(data["version"])

    def update_container(self, name: str, version: str):
        """ Keep, update, or generate data container file depending on its existence and version number. """

        # Generate container if it does not exist yet
        if not Vault.vault_path(name).exists():
            self.generate_container()
            return

        # Data container version is already up-to-date
        dc_version = Vault.container_version(name)
        if dc_version == version:
            return

        # Container version must never exceed the current one
        if dc_version > version:
            raise VersionError(f"Config container version {dc_version} > current version {version}!")

        # Metadata update is sufficient for patch level difference
        if dc_version.same_release(version):
            dc = Vault.read_container(name)
            try:
                self.generate_container(dc)
            except VersionError as exc:
                raise VersionError("Data modified, this is no patch!") from exc
            return

        # New container must be generated for minor or major level difference
        self.generate_container()


# Global interface
container_vault = Vault()
