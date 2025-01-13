"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""

from pathlib import Path

import setuptools

readme_path = Path("README.md")
with open(readme_path.absolute().as_posix(), "r") as fh:
    long_description = fh.read()

requirements_path = Path("requirements.txt")
with open(requirements_path.absolute().as_posix(), "r") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="MDSuite",
    version="0.2.0",
    author="zincwarecode",
    author_email="zincwarecode@gmail.com",
    description="A postprocessing tool for molecular dynamics simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zincware/MDSuite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["form_fac_coeffs.csv"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=required,
)
