from pathlib import Path

import compreq as cr


def set_python_version_in_github_actions(comp_req: cr.CompReq) -> None:
    python_release_set = comp_req.resolve_release_set("python", cr.python_specifier())
    minor_versions = sorted(
        {cr.floor(cr.MINOR, r.version, keep_trailing_zeros=False) for r in python_release_set},
    )
    default_version = min(minor_versions)
    minor_versions_str = ", ".join(f'"{v}"' for v in minor_versions)
    default_version_str = str(default_version)

    for yaml_path in Path(".github/workflows").glob("*.yml"):
        with cr.TextReFile.open(yaml_path) as ref:
            ref.sub(r"(^ +python-version: \")\d+\.\d+(\")$", rf"\g<1>{default_version_str}\g<2>")
            ref.sub(r"(^ +matrix:\s^ +python: \[).*(\]$)", rf"\g<1>{minor_versions_str}\g<2>")

    Path(".python-version").write_text(default_version_str)


def set_python_version(comp_req: cr.CompReq, pyproject: cr.PyprojectFile) -> cr.CompReq:
    comp_req = comp_req.for_python(
        cr.version("<", cr.ceil_ver(cr.MAJOR, cr.max_ver()))
        & cr.version(">=", cr.floor_ver(cr.MINOR, cr.max_ver(cr.min_age(years=3)))),
    )

    pyproject.set_requires_python(comp_req, cr.python_specifier())
    pyproject.set_python_classifiers(comp_req)
    set_python_version_in_github_actions(comp_req)
    version = comp_req.resolve_version("python", cr.default_python())

    tool = pyproject.toml["tool"]
    tool["mypy"]["python_version"] = f"{version.major}.{version.minor}"

    return comp_req


def main() -> None:
    with cr.PyprojectFile.open() as pyproject:
        prev_python_specifier = cr.get_bounds(pyproject.get_requires_python()).lower_specifier_set()
        comp_req = cr.CompReq(python_specifier=prev_python_specifier)
        comp_req = set_python_version(comp_req, pyproject)

        default_range = cr.version(
            ">=",
            cr.floor_ver(
                cr.REL_MINOR,
                cr.minimum_ver(
                    cr.max_ver(cr.min_age(years=1)),
                    cr.min_ver(cr.count(cr.MINOR, 3)),
                ),
            ),
        ) & cr.version("<", cr.ceil_ver(cr.REL_MAJOR, cr.max_ver()))
        dev_range = cr.version(">=", cr.floor_ver(cr.REL_MINOR, cr.max_ver())) & cr.version(
            "<",
            cr.ceil_ver(cr.REL_MINOR, cr.max_ver()),
        )

        extras = {
            "numpy": ["numpy"],
        }
        all_extra_names = {name for names in extras.values() for name in names}
        requirements = comp_req.resolve_requirement_set(
            cr.consistent_lower_bounds(
                [cr.dist(name) & default_range for name in sorted(all_extra_names)],
            ),
        )

        pyproject.set_requirements(
            comp_req,
            {n: r for n, r in requirements.requirements.items() if n not in all_extra_names},
        )
        for extra, extra_names in extras.items():
            pyproject.set_requirements(
                comp_req, {n: requirements.requirements[n] for n in extra_names}, extra=extra
            )
        pyproject.set_requirements(
            comp_req,
            [
                cr.dist("jbisect[numpy]"),
                cr.dist("compreq") & dev_range,
                cr.dist("mypy") & dev_range,
                cr.dist("pytest") & dev_range,
                cr.dist("ruff") & dev_range,
                cr.dist("taskipy") & dev_range,
                cr.dist("tomlkit") & dev_range,
            ],
            group="dev",
        )

        print(pyproject)


if __name__ == "__main__":
    main()
