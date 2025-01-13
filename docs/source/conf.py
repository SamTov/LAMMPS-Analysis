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

import sys

import sphinx_rtd_theme

sys.path.insert(0, "../..")

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Experiment information -----------------------------------------------------

project = "MDSuite"
copyright = "2021, Zincwarecode"
author = "Zincwarecode"

# The full version, including alpha/beta/rc tags
release = "2021"

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]

# Avoid name clashing
autosectionlabel_prefix_document = True

# Bibtex files for bibliography

bibtex_bibfiles = ["refs.bib"]

# Generate the API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

autodoc_default_flags = [
    "members",
    "private-members",
    "inherited-members",
    "show-inheritance",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "images/MDSUITE.png"
html_favicon = "images/MDSUITE.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "relations.html",  # needs 'show_related': True theme option to display
        "searchbox.html",
    ]
}


# -- Options for LaTeX output ---------------------------------------------

# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'letterpaper',
#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',
#     # Additional stuff for the LaTeX preamble.
#     #
#     # 'preamble': '',
#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
# }
#
# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     (master_doc, "MDSuite.tex", "MDSuite Documentation", "Contributors", "manual"),
# ]
#
#
# # -- Options for manual page output ---------------------------------------
#
# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
# man_pages = [(master_doc, "MDSuite", "MDSuite Documentation", [author], 1)]
#
#
# # -- Options for Texinfo output -------------------------------------------
#
# # Grouping the document tree into Texinfo files. List of tuples
# # (source start file, target name, title, author,
# #  dir menu entry, description, category)
# texinfo_documents = [
#     (
#         master_doc,
#         "MDSuite",
#         "MDSuite Documentation",
#         author,
#         "MDSuite",
#         "Package to analyse MD Simulations ",
#         "Miscellaneous",
#     ),
# ]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
}
