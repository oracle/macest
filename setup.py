# Copyright (c) 2021, Oracle and/or its affiliates.  All rights reserved.
# This software is licensed to you under the Universal Permissive License (UPL) 1.0 as shown at
# https://oss.oracle.com/licenses/upl
"""Setup for MACEst."""
from pathlib import Path

from setuptools import find_packages, setup

# PROJECT SPECIFIC
NAME = "macest"
PACKAGES = find_packages(where="src/")
REQUIRED = Path("requirements.txt").read_text().splitlines()
TEST_REQUIRED = Path("test_requirements.txt").read_text().splitlines()
EXTRAS = {'tests': [TEST_REQUIRED]}


version = Path("src/macest/__version__.py").read_text().split()[-1].strip('"')

setup(
    name=NAME,
    author="Mathew Rowe @ Oracle",
    author_email="matthew.r.rowe@oracle.com",
    url="https://github.com/oracle/macest",
    version=version,
    packages=PACKAGES,
    package_dir={"": "src"},
    license='??',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=REQUIRED,
    extras_requires=EXTRAS,
)
