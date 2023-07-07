"""Sphinx configuration."""
project = "Tissue"
author = "David Fischer, Mayar Ali"
copyright = "2023, David Fischer, Mayar Ali"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
