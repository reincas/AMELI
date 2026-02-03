##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
import io
import os
import platform
import zipfile

import requests
from pathlib import Path

from scidatacontainer import Container
from ameli.vault import VAULT_PATH

__version__ = "1.0.0"


def swap_name(name):
    """ Convert 'Firstname Lastname' or 'Firstname Middle Lastname' to 'Lastname, Firstname Middle'. """

    parts = name.strip().split()
    if len(parts) <= 1:
        return name
    firstname = parts[-1]
    given_names = " ".join(parts[:-1])
    return f"{firstname}, {given_names}"


class ZenodoBucket:

    bucket_url: str
    zenodo_url: str
    deposition_id: int

    def __init__(self, title, desc, sandbox=False):
        self.title = title
        self.desc = desc
        self.sandbox = sandbox

        if self.sandbox:
            self.url = "https://sandbox.zenodo.org/api/deposit/depositions"
            self.token_key = "sandbox_token"
        else:
            self.url = "https://zenodo.org/api/deposit/depositions"
            self.token_key = "token"

        self.credentials_path = self.get_cred_path()
        self.token = self.load_value(self.credentials_path, self.token_key)

        self.scidata_path = self.get_scidata_path()
        self.author = swap_name(self.load_value(self.scidata_path, "author"))
        self.email = self.load_value(self.scidata_path, "email")
        self.orcid = self.load_value(self.scidata_path, "orcid")
        self.affiliation = self.load_value(self.scidata_path, "organization")

        self.create_bucket()

    def get_cred_path(self):
        """ Return the platform-specific path to the Zenodo credentials file. """

        if platform.system() == "Windows":
            return Path(os.environ["USERPROFILE"]) / "zenodo.cfg"
        return Path.home() / ".zenodo"

    def get_scidata_path(self):
        """ Return the platform-specific path to the scidata config file. """

        if platform.system() == "Windows":
            return Path(os.environ["USERPROFILE"]) / "scidata.cfg"
        return Path.home() / ".scidata"

    @staticmethod
    def load_value(file: Path, key: str):
        """ Read string value for the given key from the given file. """

        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")
        with open(file, 'r') as f:
            for line in f:
                if line.startswith(key):
                    _, _, value = line.partition('=')
                    return value.strip()
        raise ValueError(f"Key '{key}' not found in {file}")

    def create_bucket(self):
        """ Create a Zenodo record bucket. """

        print(f"Creating Zenodo bucket '{self.title}'")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        data = {
            "metadata": {
                "upload_type": "dataset",
                "version": __version__,
                "title": self.title,
                "description": self.desc,
                "access": {"record": "public", "files": "public"},
                "license": "cc-by-sa-4.0",
                "creators": [{
                    "name": self.author,
                    "orcid": self.orcid,
                    "affiliation": self.affiliation,
                }]
            }
        }

        r = requests.post(self.url, json=data, headers=headers)
        r.raise_for_status()
        self.bucket_url = r.json()["links"]["bucket"]
        self.zenodo_url = r.json()["links"]["html"]
        self.deposition_id = r.json()["id"]
        print(f"Bucket URL: {self.bucket_url}")

    def upload_file(self, file, filename=None):
        """ Upload a file into the Zenodo bucket. """

        opened_here = False

        # File represented by file object
        if hasattr(file, 'read'):
            if filename is None:
                raise ValueError("No filename for file object.")
            if hasattr(file, 'seek'):
                file.seek(0)
            data = file

        # File represented by path
        else:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            filename = filename or path.name
            data = open(path, 'rb')
            opened_here = True

        # Upload file
        headers = {"Authorization": f"Bearer {self.token}"}
        upload_url = f"{self.bucket_url}/{filename}"
        try:
            response = requests.put(upload_url, data=data, headers=headers)
            response.raise_for_status()
            print(f"Uploaded file '{filename}' to bucket")
            return response.json()
        finally:
            if opened_here:
                data.close()

    def upload_zip(self, files, zipname):
        """ Creating a zip folder containing the given local files in the Zenodo bucket. """

        # Create zip folder in memory
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file in files:
                    zip_file.write(file, arcname=Path(file).name)

            # Upload zip folder to the bucket
            return self.upload_file(zip_buffer, filename=zipname)
