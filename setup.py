###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

from setuptools import setup
from setuptools import find_packages

import xanesnet

###############################################################################
#################################### SETUP ####################################
###############################################################################

setup(
    name = 'xanesnet',
    version = xanesnet.__version__,
    author = 'Conor D. Rankine',
    author_email = 'conor.rankine@newcastle.ac.uk',
    url = 'https://gitlab.com/conor.rankine/xanesnet',
    description = 'XANES spectrum prediction using AI/ML',
    licence = 'GPL',
    packages = find_packages(),
    install_requires = [
        'tensorflow>=2.1.0',
        'numpy',
        'scipy',
        'scikit-learn',
        'ase',
        'tqdm',
    ]
)