from pathlib import Path

import compreq as cr


def set_python_version_in_github_actions(comp_req: cr.CompReq) -> None:
    python_release_set = comp_req.resolve_release_set("python", cr.python_specifier())
    minor_versions = sorted(
        set(cr.floor(cr.MINOR, r.version, keep_trailing_zeros=False) for r in python_release_set)
    )
    default_version = min(minor_versions)
    minor_versions_str = ", ".join(f'"{v}"' for v in minor_versions)
    default_version_str = str(default_version)

    for yaml_path in Path(".github/workflows").glob("*.yml"):
        with cr.TextReFile.open(yaml_path) as ref:
            ref.sub(r"(^ +python-version: \")\d+\.\d+(\")$", rf"\g<1>{default_version_str}\g<2>")
            ref.sub(r"(^ +matrix:\s^ +python: \[).*(\]$)", rf"\g<1>{minor_versions_str}\g<2>")


def set_python_version(comp_req: cr.CompReq, pyproject: cr.PoetryPyprojectFile) -> cr.CompReq:
    comp_req = comp_req.for_python(
        cr.version("<", cr.ceil_ver(cr.MAJOR, cr.max_ver()))
        & cr.version(">=", cr.floor_ver(cr.MINOR, cr.max_ver(cr.min_age(years=5))))
    )

    pyproject.set_python_classifiers(comp_req)
    set_python_version_in_github_actions(comp_req)
    version = comp_req.resolve_version("python", cr.default_python())

    tool = pyproject.toml["tool"]
    tool["isort"]["py_version"] = int(f"{version.major}{version.minor}")
    tool["black"]["target-version"] = [f"py{version.major}{version.minor}"]
    tool["mypy"]["python_version"] = f"{version.major}.{version.minor}"

    return comp_req


def main() -> None:
    with cr.PoetryPyprojectFile.open() as pyproject:
        prev_python_specifier = cr.get_bounds(
            pyproject.get_requirements()["python"].specifier
        ).lower_specifier_set()
        comp_req = cr.CompReq(python_specifier=prev_python_specifier)
        comp_req = set_python_version(comp_req, pyproject)

        dev_range = cr.version(">=", cr.floor_ver(cr.REL_MINOR, cr.max_ver())) & cr.version(
            "<", cr.ceil_ver(cr.REL_MINOR, cr.max_ver())
        )

        pyproject.set_requirements(
            comp_req,
            [
                cr.dist("python") & cr.python_specifier(),
            ],
        )
        pyproject.set_requirements(
            comp_req,
            [
                cr.dist("black") & dev_range,
                cr.dist("compreq") & dev_range,
                cr.dist("isort") & dev_range,
                cr.dist("mypy") & dev_range,
                cr.dist("pylint") & dev_range,
                cr.dist("pytest") & dev_range,
                cr.dist("taskipy") & dev_range,
                cr.dist("tomlkit") & dev_range,
            ],
            "dev",
        )

        print(pyproject)


if __name__ == "__main__":
    main()
