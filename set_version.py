import argparse
import re
from typing import Any

import tomlkit
from packaging.version import Version

PYPROJECT_FILE = "pyproject.toml"
INIT_FILE = "jbisect/__init__.py"
INIT_VERSION_RE = re.compile(r"(^__version__ *= *\").*(\"$)")


def set_pyproject_version(version: Version) -> None:
    with open(PYPROJECT_FILE, "rt", encoding="utf-8") as fin:
        config: Any = tomlkit.load(fin)

    config["tool"]["poetry"]["version"] = str(version)

    with open(PYPROJECT_FILE, "wt", encoding="utf-8") as fout:
        tomlkit.dump(config, fout)


def set_init_version(version: Version) -> None:
    with open(INIT_FILE, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()

    lines = [INIT_VERSION_RE.sub(rf"\g<1>{version}\g<2>", line) for line in lines]

    with open(INIT_FILE, "wt", encoding="utf-8") as fout:
        fout.write("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the project version in pyproject.toml.")
    parser.add_argument("version", type=Version, help="The version to set to.")
    args = parser.parse_args()
    set_pyproject_version(args.version)
    set_init_version(args.version)


if __name__ == "__main__":
    main()
