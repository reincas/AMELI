##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import pprint
import requests

from ameli.lanthanide.content import get_configs
from ameli.lanthanide.zenodo import GitHubConfig, zenodo_lanthanide

SANDBOX = True
RECORDS = {
    #1: {"concept_id": None, "version": "1.0.0"},
    1: {"concept_id": 440211, "version": "2.0.0"},
}

GITHUB_REPO = "reincas/ameli"
GITHUB_TOKEN = GitHubConfig()["ameli_token"]

def github():
    headers_gh = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    gh_api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    response = requests.get(gh_api_url, headers=headers_gh)
    response.raise_for_status()
    data = response.json()
    release_name = data["tag_name"]
    assets = [
        {'name': f"{data['tag_name']}.zip", 'url': data['zipball_url']},
        {'name': f"{data['tag_name']}.tar.gz", 'url': data['tarball_url']}
    ]
    return release_name, assets


if __name__ == "__main__":

    for num_electrons in [1]:  # get_configs():
        version = RECORDS[num_electrons]["version"]
        concept_id = RECORDS[num_electrons]["concept_id"]
        zenodo_lanthanide(num_electrons, version, concept_id, SANDBOX)