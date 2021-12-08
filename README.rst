|madewithpython| |build| |docs| |license| |coverage|

Introduction
------------

MDSuite is a software designed specifically for the molecular dynamics community to
enable fast, reliable, and easy-to-run calculations from simulation data.

If you want to start learning about the code, you can check out the docs
`here <https://mdsuite.readthedocs.io/en/latest/>`_.

Installation
============

There are several way to install MDSuite depending on what you would like from it.

One can simply installing using

.. code-block:: bash

   pip install mdsuite

If you would like to install it from source then you can clone the repository by running

.. code-block:: bash

   git clone https://github.com/SamTov/MDSuite.git

Once you have cloned the repository, depending on whether you prefer conda or straight
pip, you should follow the instructions below.

Installation with pip
*********************

.. code-block:: bash

   cd MDSuite
   pip install .


Installation with conda
***********************

.. code-block:: bash

   cd MDSuite
   conda create -n MDSuite python=3.8
   conda activate MDSuite
   pip install .

Documentation
=============

There is a live version of the documentation hosted
`here <https://mdsuite.readthedocs.io/en/latest/>`_.
If you would prefer to have a local copy, it can be built using sphinx by following the
instructions below.

.. code-block:: bash

   cd MDSuite/docs
   make html
   cd build/html
   firefox index.html

.. badges

.. |madewithpython| image:: https://img.shields.io/badge/Made%20With-Python-blue.svg?style=flat
    :alt: Made with Python

.. |build| image:: https://github.com/zincware/MDSuite/actions/workflows/pytest.yml/badge.svg
    :alt: Build tests passing
    :target: https://github.com/zincware/MDSuite/blob/readme_badges/

.. |docs| image:: https://readthedocs.org/projects/mdsuite/badge/?version=latest&style=flat
    :alt: Build tests passing
    :target: https://mdsuite.readthedocs.io/en/latest/

.. |license| image:: https://img.shields.io/badge/License-EPLv2.0-purple.svg?style=flat
    :alt: Project license

.. |coverage| image:: https://coveralls.io/repos/github/zincware/MDSuite/badge.svg?branch=main
    :alt: Coverage Report
    :target: https://coveralls.io/github/zincware/MDSuite?branch=main
