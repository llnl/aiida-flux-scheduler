"""
Sphinx configuration for the project documentation.
"""

from importlib.metadata import PackageNotFoundError, version as package_version

project = 'AiiDA Flux Scheduler'
author = 'Nathan Keilbart'

try:
    release = package_version('aiida_flux_scheduler')
except PackageNotFoundError:
    release = '0.0.0'

version = release

extensions = [
    'myst_parser',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'sphinx_click',
    'autoapi.extension',
]

source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_heading_anchors = 3
myst_enable_extensions = [
    'colon_fence',
]

autoapi_type = 'python'
autoapi_dirs = ['../src/aiida_flux_scheduler']
autoapi_root = 'autoapi'
autoapi_add_toctree_entry = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]

html_theme = 'sphinx_book_theme'
html_title = project
html_theme_options = {
    'repository_url': 'https://github.com/LLNL/aiida-flux-scheduler',
    'use_repository_button': True,
    'use_issues_button': True,
}
