import os
import sys
from setuptools import setup, find_packages

# Add src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

setup(
    name="text_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)