# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _requirements_file(path: Path) -> set[str]:
    return {
        str(Requirement(line))
        for raw_line in path.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


def _pep517_build_requirements() -> set[str]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {str(Requirement(item)) for item in pyproject["build-system"]["requires"]}


def _pinned_base_versions(path: Path, package: str) -> set[str]:
    versions: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        if requirement.name != package:
            continue
        versions.update(
            Version(specifier.version).base_version
            for specifier in requirement.specifier
            if specifier.operator == "=="
        )
    return versions


def test_empty_build_tools_mirror_pep517_without_torch():
    pep517_without_torch = {
        requirement
        for requirement in _pep517_build_requirements()
        if Requirement(requirement).name != "torch"
    }
    empty_build_tools = _requirements_file(ROOT / "requirements/build/empty.txt")

    assert empty_build_tools == pep517_without_torch


def test_pep517_torch_requirement_remains_platform_agnostic():
    torch_requirements = {
        requirement
        for requirement in _pep517_build_requirements()
        if Requirement(requirement).name == "torch"
    }

    assert torch_requirements == {"torch==2.11.0"}


def test_cpu_build_runtime_and_test_torch_versions_match():
    requirement_files = (
        ROOT / "requirements/build/cpu.txt",
        ROOT / "requirements/cpu.txt",
        ROOT / "requirements/test/cpu.txt",
    )

    assert {
        path.relative_to(ROOT): _pinned_base_versions(path, "torch")
        for path in requirement_files
    } == {
        Path("requirements/build/cpu.txt"): {"2.11.0"},
        Path("requirements/cpu.txt"): {"2.11.0"},
        Path("requirements/test/cpu.txt"): {"2.11.0"},
    }


def test_cpu_test_image_installs_local_wheel_without_resolving_dependencies():
    dockerfile = (ROOT / "docker/Dockerfile.cpu").read_text()

    assert "uv pip install --no-deps dist/*.whl" in dockerfile
