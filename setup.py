# -*- coding: utf-8 -*-

from os import path
import setuptools
import datetime
from pathlib import Path

today = datetime.date.today().strftime("%b-%d-%Y")

# define constants
INSTALL_REQUIRES = (Path(__file__).parent / "requirements.txt").read_text().splitlines()

readme_file = Path(__file__).parent / "README.md"
readme = readme_file.read_text(encoding="utf-8")

setuptools.setup(
    name='pflow',
    author="Jiahao Fan",
    use_scm_version={'write_to': 'pflow/_version.py'},
    author_email="jiahaofan@pku.edu.cn",
    description="Rectified flow for protein folding",
    setup_requires=['setuptools_scm'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/PKUfjh/Protein_flow",
    python_requires=">=3.6",
    packages=[
        "pflow",
        "pflow/entrypoint",
        "pflow/flow",
        "pflow/common",
        "pflow/common/gromacs",
        "pflow/common/lammps",
        "pflow/common/sampler",
        "pflow/common/plumed",
        "pflow/select",
        "pflow/op",
        "pflow/superop",
        "pflow/task",
        "pflow/utils",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    package_data={'pflow/template': ['*.json', '*.sh', '*.mdp']},
    include_package_data=True,
    keywords='rectified flow protein folding pflow',
    install_requires=INSTALL_REQUIRES,
    
    entry_points={
        "console_scripts": [
            "pflow = pflow.entrypoint.main:main"
        ]
    },
)
