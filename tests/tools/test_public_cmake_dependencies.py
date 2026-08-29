# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_cmake_dependencies_do_not_require_ssh_credentials():
    cmake_sources = [REPO_ROOT / "CMakeLists.txt", *REPO_ROOT.glob("cmake/**/*.cmake")]
    credentialed_urls = []

    for source in cmake_sources:
        for line_number, line in enumerate(source.read_text().splitlines(), start=1):
            if "GIT_REPOSITORY" in line and "git@github.com:" in line:
                credentialed_urls.append(
                    f"{source.relative_to(REPO_ROOT)}:{line_number}"
                )

    assert not credentialed_urls, (
        "Public CMake dependencies must use HTTPS so container builds do not "
        f"require host SSH credentials: {credentialed_urls}"
    )
