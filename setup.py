
from setuptools import setup, find_packages

VERSION = "0.1alpha"
ISRELEASED = False

#write_version_py(VERSION, ISRELEASED, 'gui/version.py')

setup(
    name="conformation",
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    platforms=["Linux", "Unix"],
    author="Evan N. Feinberg",
    author_email="enf@stanford.edu",
    description="ConformationPy",
    license="GPL-3.0",
)

