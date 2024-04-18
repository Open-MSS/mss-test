# -*- coding: utf-8 -*-
#
# MSS - Mission Support System documentation build configuration file, created by
# sphinx-quickstart on Wed Jun 22 10:03:19 2016.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import os
import sys
import logging
import setuptools
import subprocess
import requests
import zipfile
import shutil

from string import Template

def get_tutorial_images():
    TUTORIAL_URL = "https://fz-juelich.sciebo.de/s/7DUjGMgP1HFvakG/download"
    TUTORIAL_DIR = 'videos/gif'
    if not os.path.exists(TUTORIAL_DIR):
        os.makedirs(TUTORIAL_DIR)
    TUTORIAL_ARCHIVE = 'videos/gif/tutorials.zip'
    if not os.path.exists(TUTORIAL_ARCHIVE):
        response = requests.get(TUTORIAL_URL)
        open(TUTORIAL_ARCHIVE, "wb").write(response.content)
        with zipfile.ZipFile(TUTORIAL_ARCHIVE) as zip_file:
            for item in zip_file.namelist():
                filename = os.path.basename(item)
                if not filename:
                    continue
                source = zip_file.open(item)
                target = open(os.path.join(TUTORIAL_DIR, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
        # remove zip archive
        os.remove(TUTORIAL_ARCHIVE)


get_tutorial_images()


if os.getenv("PROJ_LIB") is None or os.getenv("PROJ_LIB") == "PROJ_LIB":
    conda_file_dir = setuptools.__file__
    conda_dir = conda_file_dir.split('lib')[0]
    proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
    if "win" in sys.platform:
        proj_lib = os.path.join(os.path.join(conda_dir, 'Library'), 'share')
    os.environ["PROJ_LIB"] = proj_lib
    if not os.path.exists(proj_lib):
        os.makedirs(proj_lib)
        epsg_file = os.path.join(proj_lib, 'epsg')
        if not os.path.exists(epsg_file):
            with open(os.path.join(proj_lib, 'epsg'), 'w') as fid:
                fid.write("# Placeholder for epsg data")

if os.environ.get("GALLERY", "True") != "False":
    # Generate plot gallery
    import fs
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from mslib.mswms.demodata import DataFiles

    root_fs = fs.open_fs("~/")
    if not root_fs.exists("mss/testdata"):
        root_fs.makedirs("mss/testdata")

    examples = DataFiles(data_fs=fs.open_fs("~/mss/testdata"),
                         server_config_fs=fs.open_fs("~/mss"))
    examples.create_server_config(detailed_information=True)
    examples.create_data()

    sys.path.insert(0, os.path.join(os.path.expanduser("~"), "mss"))

    import mslib.mswms.wms
    import mslib.mswms.gallery_builder
    import importlib

    # Generate template plots
    from docs.gallery.plot_examples import HS_template, VS_template
    from mslib.mswms.mpl_lsec_styles import LS_DefaultStyle
    dataset = [next(iter(mslib.mswms.wms.mswms_settings.data))]
    mslib.mswms.wms.mswms_settings.register_horizontal_layers = [(HS_template.HS_Template, dataset)]
    mslib.mswms.wms.mswms_settings.register_vertical_layers = [(VS_template.VS_Template, dataset)]
    mslib.mswms.wms.mswms_settings.register_linear_layers = [(LS_DefaultStyle, dataset)]
    mslib.mswms.wms.server.__init__()
    mslib.mswms.wms.server.generate_gallery(sphinx=True, create=True, clear=True, simple_naming=True)
    importlib.reload(mslib.mswms.gallery_builder)

    # Generate all other plots
    mslib.mswms.wms.server.generate_gallery(sphinx=True, generate_code=True, all_plots=True, levels="3,4,200,300",
                                            vtimes="2012-10-18T00:00:00,2012-10-19T00:00:00")

# readthedocs has no past.builtins
try:
    from past.builtins import execfile
except ImportError as ex:
    logging.error("%s", ex)
execfile('../mslib/version.py')

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_rtd_theme']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'MSS - Mission Support System'
copyright = \
    '2016-2024 by the MSS team, see AUTHORS, ' \
    '2011-2014 Marc Rautenhaus, ' \
    '2008-2014 Deutsches Zentrum fuer Luft- und Raumfahrt e.V.'

author = 'see AUTHORS file'
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __version__.split('-')[0]
# The full version, including alpha/beta/rc tags.
release = __version__

# Replace $variables in the .rst files if on a readthedocs worker
if "/home/docs/checkouts" in " ".join(sys.argv):
    mss_search = subprocess.run(["conda", "search", "-c", "conda-forge", "mss"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf8").stdout
    # mss_search is inside a code block, reflect indentation
    mss_search = (" " * 3).join([line for line in mss_search.splitlines(True) if line.startswith("mss ")][-2:])

    for file in os.listdir():
        if file.endswith(".rst"):
            with open(file, "r") as rst:
                content = Template(rst.read())
            with open(file, "w") as rst:
                rst.write(content.safe_substitute(mss_version=version, mss_search=mss_search))

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'
today_fmt = '%Y-%m-%d'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_style = 'css/mss.css'
else:
    htmls_static_path = ['_static']
    html_css_files = ['mss.css']
    html_context = {
        'display_github': False, # Add 'Edit on Bitbucket' link instead of 'View page source'
        'last_updated': True,
        'commit': False,
    }

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#html_title = u'MSS - Mission Support System v1.1'

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "mss-logo.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['mss_theme', 'gallery/plots', 'videos']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
#html_last_updated_fmt = None

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr', 'zh'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# 'ja' uses this config value.
# 'zh' user can custom change `jieba` dictionary path.
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'MSS-MissionSupportSystemdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
#'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'MSS-MissionSupportSystem.tex', 'MSS - Mission Support System Documentation',
     u'A.Hilboll, A.Lal, C.Rolf, I.Krisch, J.-U.Grooß, J.Ungermann, M.Rautenhaus, R.Bauer, S.Padhi, T.Breuer', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'mss-missionsupportsystem', 'MSS - Mission Support System Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'MSS-MissionSupportSystem', 'MSS - Mission Support System Documentation',
     author, 'MSS-MissionSupportSystem', 'One line description of project.',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False
suppress_warnings = ['image.nonlocal_uri']
