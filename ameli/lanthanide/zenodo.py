##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
import io
import os
import platform
import re
import requests
import zipfile
from pathlib import Path

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

        self.meta = {
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

        self.create_bucket()
        self.url = f"{self.url}/{self.deposition_id}"

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
        data = {"metadata": self.meta}

        response = requests.post(self.url, json=data, headers=headers)
        response.raise_for_status()
        self.bucket_url = response.json()["links"]["bucket"]
        self.zenodo_url = response.json()["links"]["html"]
        self.deposition_id = response.json()["id"]
        print(f"Bucket {self.deposition_id}: created")

    def update_meta(self, **kwargs):
        data = {"metadata": self.meta | kwargs}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"}
        response = requests.put(self.url, json=data, headers=headers)
        response.raise_for_status()
        print(f"Bucket {self.deposition_id}: Metadata updated")
        return response.json()

    def upload_file(self, file, filename=None):
        """ Upload a file into the Zenodo bucket. """

        opened_here = False

        # File represented by file object
        if hasattr(file, 'read'):
            assert filename, "No filename for file object."
            if hasattr(file, 'seek'):
                file.seek(0)
            data = file

        # Text file represented by text string
        elif filename:
            data = io.BytesIO(file.encode('utf-8'))
            opened_here = True

        # File represented by path
        else:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            filename = path.name
            data = open(path, 'rb')
            opened_here = True

        # Upload file
        headers = {"Authorization": f"Bearer {self.token}"}
        upload_url = f"{self.bucket_url}/{filename}"
        try:
            response = requests.put(upload_url, data=data, headers=headers)
            response.raise_for_status()
            print(f"Bucket {self.deposition_id}: Uploaded file '{filename}'")
            return response.json()
        finally:
            if opened_here:
                data.close()

    def upload_zip(self, files, root_path, zip_name):
        """ Creating a zip folder containing the given local files in the Zenodo bucket. Take zip internal file paths
        relative to the given root path. """

        # Create zip folder in memory
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file in files:
                    arcname = Path(file).relative_to(root_path)
                    zip_file.write(file, arcname=arcname)

            # Upload zip folder to the bucket
            return self.upload_file(zip_buffer, filename=zip_name)


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


def get_zip_folders(num_electrons):
    """ Generate names, configuration root path, and file lists for zip folders containing all data container files
    available for the lanthanide ion with the given number of electrons. """

    # Prepare root folder of the given lanthanide configuration
    config_name = f"f{num_electrons}"
    root_path = VAULT_PATH / Path(config_name)

    # Content mapping for each zip folder
    zip_structure = [
        ("product.zip", [Path("product")]),
        ("sljm.zip", [Path("sljm")]),
        ("slj.zip", [Path("slj")]),
        ("slj_reduced.zip", [Path("slj_reduced")]),
        ("support.zip", [Path("."), Path("unit")]),
    ]

    # Generate list of data container files for each zip folder in the given order
    for zip_name, subfolders in zip_structure:
        files = []
        for folder in subfolders:
            search_dir = root_path / folder
            assert search_dir.is_dir(), f"Folder '{folder}' does not exist!"
            for file_path in search_dir.glob("*.zdc"):
                files.append(file_path)
        yield zip_name, root_path, files
