# To install project src into project virtual environment from command line:
# > pip install -e .

# to uninstall use:
# > pip uninstall [package_name]

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    packages=find_packages(),
)