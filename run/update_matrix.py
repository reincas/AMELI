import json
import zipfile
import pathlib
import tempfile
import shutil
import re
from ameli.vault import VAULT_PATH
from ameli.matrix import MatrixName


def update_matrix(file: pathlib.Path):
    # Define the internal path to the target file
    internal_path = 'data/matrix.json'

    # Preliminary Check
    with zipfile.ZipFile(file, 'r') as zin:
        if internal_path not in zin.namelist():
            return  # Or raise an error if the file must exist

        with zin.open(internal_path) as f:
            data = json.load(f)

        # Check if update is actually necessary
        has_rank = "tensorRank" in data
        has_type = "elementType" in data

        if has_rank and has_type:
            print(f"{file} is up-to-date")
            return

    # Create a temporary directory to handle the swap
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_path = pathlib.Path(temp_dir) / "temp.zip"

        with zipfile.ZipFile(file, 'r') as zin:
            with zipfile.ZipFile(temp_zip_path, 'w') as zout:
                # Copy all files from original to new, except the one we're changing
                for item in zin.infolist():
                    if item.filename == internal_path:
                        # Load, modify, and write the JSON
                        with zin.open(item.filename) as f:
                            data = json.load(f)

                        name = data["name"]
                        rank = MatrixName(name).rank
                        print(f"{file}: {name} -> {rank}")
                        # Add the new elements
                        data["tensorRank"] = rank
                        data["elementType"] = "normal"

                        # Write modified JSON back to the new ZIP
                        zout.writestr(item.filename, json.dumps(data, indent=4))
                    else:
                        # Copy other files unchanged
                        zout.writestr(item, zin.read(item.filename))

        # Replace the original ZIP file with the updated one
        shutil.move(str(temp_zip_path), str(file))


def all_matrices(root_path: pathlib.Path):
    root = pathlib.Path(root_path)
    f_pattern = re.compile(r'^f[0-9]+$')
    for f_dir in root.iterdir():
        if f_dir.is_dir() and f_pattern.match(f_dir.name):
            sym_dir = f_dir / "symbolic"
            if sym_dir.is_dir():
                target_subs = ["Product", "SLJM", "SLJ", "SLJreduced"]
                for sub_name in target_subs:
                    sub_dir = sym_dir / sub_name
                    if sub_dir.is_dir():
                        yield from sub_dir.rglob("*.zdc")


for file in all_matrices(VAULT_PATH):
    update_matrix(file)
