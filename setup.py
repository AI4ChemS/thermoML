from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'Physics-informed ML for thermal fluid property prediction'
LONG_DESCRIPTION = """
ThermoML is a Python package for predicting thermophysical properties of pure fluids using chemistry- and temperature-aware machine learning models. This tool integrates physics-informed modeling with machine learning techniques to accurately predict thermophyiscal properties across temperature ranges. The package includes pre-trained models, data preprocessing utilities, and simple interfaces for inference and evaluation.

Key Features:
1. Predict property of interest (such as dynamic viscosity) from SMILES and temperature
2. Flexible equation integration based on the property of interest (e.g., Arrhenius-based scaling for viscosity)
3. Easy-to-use for batch predictions
4. Includes curated datasets and example notebooks

Whether you're working on thermal fluid research, chemical engineering, or data-driven materials science, ThermoML provides a fast and extensible way to estimate temperature-dependent fluid properties.
"""

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="thermoML",  # Package name on PyPI
    version="0.1.0",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    author="Mahyar Rajabi-Kochi",
    author_email="mahyar.rajabi@mail.utoronto.ca",
    url="https://github.com/AI4ChemS/thermoML",  # Change as needed
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "python", "chemistry-ml", "thermodynamics-informed-ml",
        "machine-learning", "thermal-fluids"
    ],
    python_requires=">=3.9",
)