import re
from pathlib import Path
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


def refactor_subfolders(num):
    root = VAULT_PATH / Path(f"f{num}")

    rename_map = {
        "Product": "product",
        "SLJM": "sljm",
        "SLJ": "slj",
        "SLJreduced": "slj_reduced"
    }

    if not root.exists():
        print(f"Directory {root} does not exist.")
        return

    for old_name, new_name in rename_map.items():
        old_path = root / old_name
        new_path = root / new_name

        if old_path.is_dir():
            if new_path.exists() and old_path.resolve() != new_path.resolve():
                print(f"Warning: Cannot rename {old_name} because {new_name} already exists.")
            else:
                old_path.rename(new_path)
                print(f"Renamed: {old_path} -> {new_path}")


if __name__ == "__main__":
    nums = get_configs()
    for num in nums:
        refactor_subfolders(num)
