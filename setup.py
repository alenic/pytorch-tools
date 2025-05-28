import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pytorch_tools",
    version = "0.0.5",
    author = "Alessandro Nicolosi",
    description = "Useful pytorch tools",
    license = "MIT",
    url = "https://github.com/alenic/pytorch-tools",
    packages=find_packages(),
    long_description=read('README.md'),
)