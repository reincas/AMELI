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

import hashlib
import io
import json
import os
import platform
import requests
import tempfile
import zipfile
from pathlib import Path

from ameli.vault import Version, VersionError

from lanthanide import LANTHANIDE_IONS
from content import get_root_path, get_zip_folders
from dataset import description


##########################################################################
# Configuration file classes
##########################################################################

def swap_name(name):
    """ Convert 'Firstname Lastname' or 'Firstname Middle Lastname' to 'Lastname, Firstname Middle'. """

    parts = name.strip().split()
    if len(parts) <= 1:
        return name
    firstname = parts[-1]
    given_names = " ".join(parts[:-1])
    return f"{firstname}, {given_names}"


class ConfigFile:
    def __init__(self, path):
        self.path = path

        self.content = {}
        with open(self.path, 'r') as f:
            for line in f:
                if line[:1] == '#':
                    continue
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                self.content[key] = value

    def __getitem__(self, key):
        return self.content[key]


class ScidataConfig(ConfigFile):
    def __init__(self):
        if platform.system() == "Windows":
            path = Path(os.environ["USERPROFILE"]) / "scidata.cfg"
        else:
            path = Path.home() / ".scidata"
        super().__init__(path)


class GitHubConfig(ConfigFile):
    def __init__(self):
        if platform.system() == "Windows":
            path = Path(os.environ["USERPROFILE"]) / "github.cfg"
        else:
            path = Path.home() / ".github"
        super().__init__(path)


class ZenodoConfig(ConfigFile):
    def __init__(self):
        if platform.system() == "Windows":
            path = Path(os.environ["USERPROFILE"]) / "zenodo.cfg"
        else:
            path = Path.home() / ".zenodo"
        super().__init__(path)


##########################################################################
# File representation classes
##########################################################################

class FileHash:
    """ This class is representing a 'hashes.json' file. """

    def __init__(self):
        self.hashes = {}

        self.hash = None
        self.sha256_data = None
        self.is_immutable = False

    def append(self, file):
        """ Append the hashes of the given file to the hash. """

        raise NotImplementedError()

    def finalize(self):
        """ Aggregate file and data hashes and make the dictionary immutable. """

        assert not self.is_immutable
        filenames = [fn for fn, value in self.hashes.items() if isinstance(value, dict)]
        for key in ("hash", "sha256Data"):
            hasher = hashlib.sha256()
            for arc_name in sorted(filenames):
                hasher.update((self.hashes[arc_name][key]).encode("utf8"))
            self.hashes[key] = hasher.hexdigest()

        self.hash = self.hashes["hash"]
        self.sha256_data = self.hashes["sha256Data"]
        self.is_immutable = True

    def bytes(self):

        assert self.is_immutable
        return json.dumps(self.hashes, sort_keys=True, indent=4).encode("utf8")


class ContainerHash(FileHash):
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        super().__init__()

    def append(self, filename: str):
        """ Append the hashes of the given data container. """

        assert not self.is_immutable
        arc_name = Path(filename).relative_to(self.root_path)
        assert arc_name.suffix == ".zdc"

        with zipfile.ZipFile(filename, 'r') as zip_file:
            with zip_file.open('content.json') as f:
                content = json.load(f)

        self.hashes[str(arc_name)] = {"hash": content["hash"], "sha256Data": content["sha256Data"]}


class RecordHash(FileHash):

    def append(self, file: "ZipFolder"):
        """ Append the hashes of the given ZIP folder. """

        assert isinstance(file, ZipFolder)
        self.hashes[file.name] = {"hash": file.hashes.hash, "sha256Data": file.hashes.sha256_data}

    def merge(self, zenodo):
        remote_files = set(fi["filename"] for fi in zenodo.record["files"] if fi["filename"] != "hashes.json")
        local_files = set(fn for fn, value in self.hashes.items() if isinstance(value, dict))
        assert local_files == remote_files

        for filename in local_files:
            file_info = zenodo.file_info(filename)
            self.hashes[filename] |= {"checksum": file_info["checksum"], "filesize": file_info["filesize"]}


class ZipFolder:
    """ Data class representing a ZIP folder containing a number of data containers and the file 'hashes.json' with
    all file and data hashes of the data containers. """

    def __init__(self, name, root_path, filenames):
        """ Store name of a ZIP folder containing the given local files for the Zenodo record. Take file paths inside
        the ZIP folder relative to the given root path. """

        self.name = name
        self.root_path = Path(root_path)
        self.filenames = filenames

        # Get file and data hashes
        self.hashes = ContainerHash(self.root_path)
        for filename in self.filenames:
            self.hashes.append(filename)
        self.hashes.finalize()

    def __str__(self):
        """ String representation of the object. """

        return f"ZipFolder({self.name})"

    def generate(self):
        """ Generate ZIP folder in the given (temporary) file object. """

        # Generate the ZIP folder
        tmp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=True)
        with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_STORED) as zip_file:
            # Store every file in the ZIP folder
            for filename in sorted(self.filenames):
                arc_name = Path(filename).relative_to(self.root_path)
                zip_file.write(filename, arcname=arc_name)

            # Store 'hashes.json'
            zip_file.writestr("hashes.json", self.hashes.bytes())

        # Return the flushed temporary file
        tmp_zip.flush()
        return tmp_zip


##########################################################################
# Zenodo record class
##########################################################################

class Zenodo:

    def __init__(self, concept_id, sandbox=True):
        """ Load or create a Zenodo record. If the attribute 'is_submitted' is False, this is a deposition record. """

        self.concept_id = concept_id
        self.is_sandbox = bool(sandbox)

        # Zenodo config file contains tokens
        self.zenodo_cfg = ZenodoConfig()

        # Get base URL and authentication token
        if self.is_sandbox:
            self.url = "https://sandbox.zenodo.org/api"
            self.token = self.zenodo_cfg["sandbox_token"]
        else:
            self.url = "https://zenodo.org/api"
            self.token = self.zenodo_cfg["token"]
        self.auth_header = {"Authorization": f"Bearer {self.token}"}

        # Get unsubmitted draft record
        if self.concept_id is None:

            # Fetch an existing unsubmitted draft record or create a new one
            self.record = self.find_deposition()
            if not self.record:
                self.record = self.create_deposition()
            self.concept_id = int(self.record["conceptrecid"])
            print(f"*** STORE CONCEPT RECORD ID: {self.concept_id} ***")

        # Load submitted record
        else:

            # Get record by concept ID
            self.record = self.get_concept(self.concept_id)

            # Discard pending draft of submitted record
            if self.record:
                draft_record = self.find_deposition(self.concept_id)
                if draft_record:
                    self.record = draft_record

            # Get existing unsubmitted draft record
            else:
                self.record = self.find_deposition(self.concept_id)
                if self.record is None:
                    raise ValueError(f"Concept ID {self.concept_id} not found on Zenodo!")

        # Load fresh record
        self.load_record()
        self.is_submitted = self.record["submitted"]

        # Sanity check
        assert self.concept_id == int(self.record["conceptrecid"])

        # Extract metadata
        self.metadata = self.record["metadata"]
        self.version = Version(self.metadata["version"]) if "version" in self.metadata else None

        # Load the hashes file
        self.hashes = self.load_hashes()

    @property
    def next_version(self):
        """ Return next version (patch level) of the record. """

        return self.version.next_version()

    @property
    def next_release(self):
        """ Return next release version (minor level) of the record. """

        return self.version.next_release()

    @property
    def next_major(self):
        """ Return next major release version (major level) of the record. """

        return self.version.next_major()

    def load_hashes(self):
        """ Load and return the 'hashes.json' file of the record. Compare checksums and file sizes of all files. """

        if not self.has_file("hashes.json"):
            return {}

        hashes = self.download_json("hashes.json")
        for filename, value in hashes.items():
            if not isinstance(value, dict):
                continue
            file_info = self.file_info(filename)
            if self.is_submitted:
                assert value["filesize"] == file_info["size"]
                assert value["checksum"] == file_info["checksum"].split(":", 1)[1]
            else:
                assert value["filesize"] == file_info["filesize"]
                assert value["checksum"] == file_info["checksum"]
        return hashes

    def find_deposition(self, concept_id=None):
        """ Try to get an unsubmitted Zenodo record. """

        url = f"{self.url}/deposit/depositions"
        params = {"size": 50, "page": 1}
        while True:
            response = requests.get(url, params=params, headers=self.auth_header)
            response.raise_for_status()
            records = response.json()
            for record in records:
                if record["state"] == "unsubmitted" \
                        and (concept_id is None or int(record["conceptrecid"]) == concept_id):
                    return record
            if "next" in response.links:
                params["page"] += 1
            else:
                break

    def create_deposition(self):
        """ Create an empty Zenodo record. """

        headers = {"Content-Type": "application/json"} | self.auth_header
        data = {"metadata": {"tite": "<no title>", "description": "<no description>", "version": "1.0.0"}}
        url = f"{self.url}/deposit/depositions"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_concept(self, concept_id):
        """ Get latest record of the given concept ID. Return None if no record is published yet. """

        url = f"{self.url}/records/{concept_id}"
        response = requests.get(url, headers=self.auth_header)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_deposition(self, record):
        """ Get pending draft of the given Zenodo record. Return None if no draft exists. """

        record_id = record["id"]
        url = f"{self.url}/deposit/depositions/{record_id}"
        response = requests.get(url, headers=self.auth_header)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def discard_record(self, record):
        """ Discard given pending draft record. """

        url = record["links"]["discard"]
        response = requests.post(url, headers=self.auth_header)
        response.raise_for_status()

    def load_record(self):
        """ Return the current Zenodo record. """

        url = self.record["links"]["self"]
        response = requests.get(url, headers=self.auth_header)
        response.raise_for_status()
        self.record = response.json()

    def update_meta(self, metadata):
        """ Update metadata of a Zenodo record and store the response. """

        headers = {"Content-Type": "application/json"} | self.auth_header
        data = {"metadata": self.metadata | metadata}
        data["metadata"].pop("resource_type", None)
        data["metadata"].pop("doi", None)

        if self.is_submitted:
            record = self.get_deposition(self.record)
            url = record["links"]["edit"]
            response = requests.post(url, headers=self.auth_header)
            response.raise_for_status()

            url = response.json()["links"]["self"]
            response = requests.put(url, json=data, headers=headers)
            response.raise_for_status()

            url = response.json()["links"]["publish"]
            response = requests.post(url, headers=self.auth_header)
            response.raise_for_status()

        else:
            url = self.record["links"]["latest_draft"]
            response = requests.put(url, json=data, headers=headers)
            response.raise_for_status()

        self.load_record()
        assert self.is_submitted == self.record["submitted"]
        assert self.concept_id == int(self.record["conceptrecid"])

    def filenames(self):
        """ Return list of all names of the files in the Zenodo record. """

        key = "key" if self.is_submitted else "filename"
        return [file_info[key] for file_info in self.record["files"]]

    def file_info(self, filename):
        """ Return the file info dictionary from the Zenodo response. """

        filenames = self.filenames()
        if filename not in filenames:
            raise FileNotFoundError(f"File '{filename}' not found in Zenodo record {self.record["id"]}")
        index = filenames.index(filename)
        return self.record["files"][index]

    def has_file(self, filename):
        """ Return True if a file exists in the Zenodo response. """

        try:
            self.file_info(filename)
        except FileNotFoundError:
            return False
        return True

    def upload_file(self, file, filename):
        """ Upload a file into the Zenodo record. """

        url = f"{self.record["links"]["bucket"]}/{filename}"
        file.seek(0)
        response = requests.put(url, data=file, headers=self.auth_header)
        file.close()
        response.raise_for_status()
        r = response.json()
        return r

    def download_json(self, filename):
        """ Download and return the given JSON file from the record. """

        # Download and return file
        file_info = self.file_info(filename)
        key = "self" if self.is_submitted else "download"
        url = file_info["links"][key]
        response = requests.get(url, headers=self.auth_header)
        response.raise_for_status()
        return response.json()

    def delete_file(self, filename):
        """ Delete given file from the Zenodo record. """

        file_info = self.file_info(filename)
        url = file_info["links"]["self"]
        response = requests.delete(url, headers=self.auth_header)
        if response.status_code == 204:
            return
        response.raise_for_status()

    def publish(self):
        """ Publish Zenodo record. """

        url = self.record["links"]["publish"]
        response = requests.post(url, headers=self.auth_header)
        response.raise_for_status()

        self.record = self.get_concept(self.concept_id)
        self.is_submitted = self.record["submitted"]
        assert self.is_submitted

        self.hashes = self.load_hashes()

    def create_release(self, version):
        """ Create a Zenodo release. """

        record = self.get_deposition(self.record)

        url = record["links"]["newversion"]
        response = requests.post(url, headers=self.auth_header)
        response.raise_for_status()
        record = response.json()

        url = record["links"]["latest_draft"]
        response = requests.get(url, headers=self.auth_header)
        response.raise_for_status()
        self.record = response.json()

        self.is_submitted = self.record["submitted"]
        assert not self.is_submitted

##########################################################################
# Upload functions for Zenodo records
##########################################################################

def upload_record(zenodo, hashes, meta, files):
    """ Upload and publish a Zenodo record. """

    # Remove all existing files from record
    for filename in zenodo.filenames():
        zenodo.delete_file(filename)

    # Update record metadata and upload all data files
    zenodo.update_meta(meta)
    for file in files:
        zenodo.upload_file(file.generate(), file.name)
    zenodo.load_record()

    # Upload hashes
    hashes.merge(zenodo)
    zenodo.upload_file(io.BytesIO(hashes.bytes()), "hashes.json")

    # Submit record
    zenodo.publish()


def upload_zenodo(version, title, desc, files, zenodo):
    """ Upload or update metadata and files as Zenodo record. """

    # Dataset version
    version = Version(version)

    # Metadata dictionary
    scidata_cfg = ScidataConfig()
    metadata = {
        "upload_type": "dataset",
        "version": str(version),
        "title": title,
        "description": desc,
        "language": "eng",
        "access_right": "open",
        "license": "cc-by-sa-4.0",
        "creators": [{
            "name": swap_name(scidata_cfg["author"]),
            "orcid": scidata_cfg["orcid"],
            "affiliation": scidata_cfg["organization"],
        }],
    }

    # Aggregate all hashes
    hashes = RecordHash()
    for file in files:
        hashes.append(file)
    hashes.finalize()

    # Zenodo record is in draft state
    if not zenodo.is_submitted:
        upload_record(zenodo, hashes, metadata, files)

    # Zenodo record is in submitted state
    else:
        release = False

        if zenodo.hashes["hash"] == hashes.hash:
            assert zenodo.hashes["sha256Data"] == hashes.sha256_data
            if version == zenodo.version:
                return
            if version != zenodo.next_version:
                raise VersionError(f"Version {zenodo.next_version} expected for metadata update, got {version}!")
        elif zenodo.hashes["sha256Data"] == hashes.sha256_data:
            if version != zenodo.next_release:
                raise VersionError(f"Version {zenodo.next_release} expected for new release, got {version}!")
            release = True
        else:
            if version != zenodo.next_major:
                raise VersionError(f"Version {zenodo.next_major} expected for new major release, got {version}!")
            release = True

        # Create new release of Zenodo record
        if release:
            zenodo.create_release(version)
            upload_record(zenodo, hashes, metadata, files)

        # Update record metadata
        else:
            zenodo.update_meta(metadata)

def zenodo_lanthanide(num_electrons, version, concept_id, sandbox=True):
    """ Upload or update Zenodo dataset record for the given lanthanide ion. """

    print(f"Electrons: {num_electrons}")
    zenodo = Zenodo(concept_id, sandbox)

    title = f"Matrix Elements for the {LANTHANIDE_IONS[num_electrons]} Ion"
    desc = description(num_electrons)
    root_path = get_root_path(num_electrons)
    files = []
    for zip_name, filenames in get_zip_folders(root_path):
        file = ZipFolder(zip_name, root_path, filenames)
        files.append(file)

    upload_zenodo(version, title, desc, files, zenodo)

