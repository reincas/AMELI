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
import os
import re
import time
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

    def write(self, name: str, items: dict):
        dc = Container(items=items)
        dc.freeze()
        path = self.vault_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        dc.write(str(tmp_path))

        max_retries = 100
        for i in range(max_retries):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                time.sleep(0.05 * (i + 1))
        raise RuntimeError(f"Storage of {path} failed!")

    def read(self, name: str) -> Container:

        path = self.vault_path(name)
        return Container(file=str(path))

    @staticmethod
    def in_vault(name: str) -> bool:
        return Vault.vault_path(name).exists()

    @staticmethod
    def vault_path(name: str) -> Path:
        """ Return Path object for the given file name. """

        return VAULT_PATH / Path(name)

    def generate_container(self, dc=None):
        raise NotImplementedError()

    def load_container(self, name: str, version: str):
        """ Load, update or generate data container file depending on its existence and version number. """

        # Generate container if it does not exist yet
        if not self.vault_path(name).exists():
            self.generate_container()

        # Load container and get its version
        dc = self.read(name)
        dc_version = Version(dc["meta.json"]["version"])

        # Container version must never exceed the current one
        if dc_version > version:
            raise VersionError(f"Config container version {dc_version} > current version {version}!")

        # Version difference to be resolved
        elif dc_version < version:

            # Container update is sufficient for patch level difference
            if dc_version.same_release(version):
                try:
                    self.generate_container(dc)
                except VersionError as exc:
                    raise VersionError("Data modified, this is no patch!") from exc

            # New container nust be generated for minor or major level difference
            else:
                self.generate_container()

            # Re-load data container
            dc = self.read(name)

        # Return data container
        return dc


# Global interface
container_vault = Vault()
