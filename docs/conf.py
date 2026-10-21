# -- SimRL documentation configuration --------------------------------
# Sphinx with pydata-sphinx-theme, MyST Markdown throughout.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information --------------------------------------------------

project = "SimRL"
copyright = "2026, SimRL contributors"
author = "SimRL contributors"
release = "0.1.0"

# -- General configuration ------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# -- MyST configuration ---------------------------------------------------

# `colon_fence` is required for the `:::{grid}` directives on the landing page.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "attrs_inline",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Annotation-heavy signatures produce a flood of unresolvable cross-reference
# warnings. Revisit once a curated API reference page exists.
nitpicky = False

# -- HTML output ----------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "SimRL Documentation"

html_theme_options = {
    "logo": {
        "image_light": "_static/img/SimRL_logo_merge.png",
        "image_dark": "_static/img/SimRL_logo_merge.png",
        "alt_text": "SimRL",
    },
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navbar_align": "left",
    # Four top-level sections: getting started, the pipeline, reference, development.
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# -- Copybutton -----------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |> "
copybutton_prompt_is_regexp = True

# -- Mermaid --------------------------------------------------------------

mermaid_output_format = "raw"

# -- Napoleon -------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
