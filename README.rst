========
thermoML
========


.. image:: https://img.shields.io/pypi/v/thermoml.svg
        :target: https://pypi.python.org/pypi/thermoml

.. image:: https://img.shields.io/travis/Mahyar-rajabi94/thermoml.svg
        :target: https://travis-ci.com/Mahyar-rajabi94/thermoml

.. image:: https://readthedocs.org/projects/thermoml/badge/?version=latest
        :target: https://thermoml.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




        Thermodynamics-Informed Machine Learning for Temperature-Dependent Fluid Properties
        ===============================================================================

        This repository accompanies the paper presenting "Thermodynamics-informed machine learning to predict temperature-dependent properties of fluids". By combining established physics-based equations, such as the Arrhenius equation, with machine learning models, this approach encodes temperature dependence directly into the predictive framework. The model predicts the chemistry-dependent coefficients of equation, enabling accurate and generalizable predictions across diverse chemistries and temperature ranges. The methodology has been validated using experimental and benchmarked against two different base models.

        .. image:: images/figure.svg
           :alt: Thermodynamics-Informed Model Framework
           :align: center

        Repository Structure
        --------------------

        **1. Jupyter Notebook**

        - **`viscosity.ipynb`**: This notebook serves as the main entry point for running the analysis. It is organized into the following sections:

          1. **Thermodynamics-Informed Model Training**: Train the thermodynamics-informed model using the provided dataset.
          2. **Thermodynamics-Informed Model Testing**: Evaluate the trained model on test data.
          3. **Multitemperature Base Model**: Train and test a base ML model using datasets spanning multiple temperatures.
          4. **Isothermal Base Model**: Train and test a base ML model using isothermal datasets.
          5. **Generating Parity Plots and Accuracy Matrices**: Compare the results and visualize the performance of the models.

        **2. Utils Folder**

        The **`utils`** folder contains core scripts and functions for building and training models:

        1. **`hp_tuning.py`**: Contains code for hyperparameter tuning using **Optuna**.
        2. **`isothermal_base_model.py`**: Implements the isothermal base model, a predictive ML model trained on isothermal data. It does not use any temperature-property equations and takes temperature as a direct input.
        3. **`multitemperature_base_model.py`**: Implements the multitemperature base model, trained on datasets covering five temperature levels. Like the isothermal model, it does not rely on equations.
        4. **`thermoML.py`**: Implements the thermodynamics-informed model, including:
           - Converting SMILES to numerical descriptors using the **MORDRED Python package**.
           - Feature selection through a pipeline (details below).
           - Training an ensemble of mANN models using **BAGGING**.
           - Performing uncertainty assessment.

        **3. Data Folder**

        - **`data`**: Contains the datasets required for training and testing the models. These include dynamic viscosity data for the fluids analyzed in the study.

        Running the Code
        ----------------

        **1. Set Up the Environment**

        - Install the required dependencies listed in **`requirements.txt`**.
        - Ensure you have Python 3.8+ installed.

        **2. Running the Analysis**

        - Open **`viscosity.ipynb`** in Jupyter Notebook.
        - Follow the sections to train, test, and compare models.

        **3. Feature Selection Pipeline**

        The thermodynamics-informed model uses a feature selection pipeline to identify the most informative descriptors:

        - **BAGGING**: Creates diverse training datasets by sampling with replacement.
        - **Ensemble Learning**: Trains an ensemble of mANN models for improved accuracy and uncertainty quantification.

        How to Reproduce the Results
        -----------------------------

        1. Use the **`viscosity.ipynb`** notebook to train and test the models as outlined.
        2. Refer to the **`thermoML.py`** script for implementing the thermodynamics-informed approach.
        3. Generate parity plots and compare accuracy matrices using the provided code.
