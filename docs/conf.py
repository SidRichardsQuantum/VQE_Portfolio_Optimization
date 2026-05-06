from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "VQE Portfolio Optimization"
author = "Sid Richards"
copyright = "2026, Sid Richards"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["portfolio.css"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

suppress_warnings = [
    "docutils",
    "misc.highlighting_failure",
    "myst.header",
    "myst.xref_missing",
]
