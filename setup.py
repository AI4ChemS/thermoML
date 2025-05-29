from setuptools import setup, find_packages
import os

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
setup(
    name="thermoML",
    version=VERSION,
    author="Mahyar Rajabi-Kochi",
    author_email="mahyar.rajabi@mail.utoronto.ca",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/AI4ChemS/thermoML",
    packages=find_packages(include=["thermoML", "thermoML.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "pubchempy>=1.0",
        "requests>=2.25",
        "mordred>=1.2.0",
        "scikit-learn>=1.0",
        "xgboost>=1.6",
        "optuna>=3.0",
        "openpyxl",
        "matplotlib"
    ],
    extras_require={
        "rdkit": ["rdkit>=2022.3.5"],  # recommend installing via conda
        "tensorflow": ["tensorflow>=2.12.0"]
    },
    keywords=[
        "python", "chemistry-ml", "thermodynamics-informed-ml",
        "machine-learning", "thermal-fluids"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)