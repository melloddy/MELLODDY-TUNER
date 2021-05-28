from os.path import realpath, dirname, join
from setuptools import setup, find_packages

import melloddy_tuner

VERSION = melloddy_tuner.utils.version.__version__
PROJECT_ROOT = dirname(realpath(__file__))


setup(
    name="melloddy_tuner",
    version=VERSION,
    author="Lukas Friedrich",
    author_email="lukas.friedrich@merckgroup.com",
    description="melloddy_tuner",
    url="https://github.com/melloddy/MELLODDY-TUNER",
    packages=find_packages(),
    package_data={
        "": ["LICENSE.txt", "README.md", "requirements.txt", "data/reference_set.csv"]
    },
    include_package_data=True,
    entry_points={"console_scripts": ["tunercli = melloddy_tuner.tunercli:main"]},
    license="MIT License",
    long_description="""
      Processing and standardization of structure and bioactivity data for federated machine learning models in drug discovery.

      MELLODDY-PROJECT
      =============
      https://www.melloddy.eu


      This tool is hosted at https://github.com/melloddy/MELLODDY-TUNER
      """,
)
