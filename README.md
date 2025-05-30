# ThermoML

This repository accompanies the paper presenting *"Thermodynamics-informed machine learning to predict temperature-dependent properties of fluids"*. By combining established physics-based equations, such as the Arrhenius equation, with machine learning models, this approach encodes temperature dependence directly into the predictive framework. The model predicts the chemistry-dependent coefficients of the equation, enabling accurate and generalizable predictions across diverse chemistries and temperature ranges. The methodology has been validated using experimental data and benchmarked against two different base models.

![Model Architecture](images/figure.svg)

## Hardware requirements
This work was done by training and deploying the models on a GPU (NVIDIA RTX4090), so it is advised to utilize a machine that aligns with similar computational powers. Regarding run-time:
1. ``demo/finetuning.ipynb``: 10 minutes to run full notebook
2. ``demo/predictions.ipynb``: A few seconds per cell

## Software requirements
This software has been tested and works on Linux (Ubuntu 22.04.4 LTS) and iOS v14.4.

## Installation
Prior to installation of this package, it is recommended to start a fresh environment. This can be achieved from following this:
```
conda create -n thermo python=3.9
conda activate thermo
```

### Pip installable version
Our package is available on PyPI, and is pip installable from simply performing:
```
pip install thermoML
```

### Developer version
```
git clone https://github.com/AI4ChemS/thermoML.git
cd path/to/thermoML
pip install -e .
```

## Usage
Demonstrations for training the thermodynamics-informed model, a model using datasets spanning multiple temperatures and a model using isothermal datasets can be found in ``demo/finetuning.ipynb``, with an additional demonstration on making predictions with each saved model found in ``demo/predictions.ipynb``. In order to summarize the different models available:
1. **Thermodynamics-Informed Model Training**: Train the thermodynamics-informed model using the provided dataset, and making predictions with it.
2. **Multitemperature Base Model**: Train and test a base ML model using datasets spanning multiple temperatures.
3. **Isothermal Base Model**: Train and test a base ML model using isothermal datasets.

For a brief demonstration on training each model, the following starter code has been provided.

```python
from thermoML import main_training
from thermoML import main_base_train_MT
from thermoML import main_base_train_ST

## Refer to package files and demo for explanations on kwargs

df, df_mu_log, df_mu =  main_training(df, arr_data, mu_temp, path, to_drop = ['Compounds', 'smiles'], **kwargs) # thermo-informed
main_base_train_MT(df, df_mu, set_size, model_path, **kwargs) # multi-temp
main_base_train_ST(df, df_mu, set_size, model_path, **kwargs) # isothermal

```

## Data Availability

The folder in `demo/data` ontains the datasets required for training and testing the models. These include dynamic viscosity data for the fluids analyzed in the study.


## Available features of interest

The **`utils`** folder contains core scripts and functions for building and training models:

1. **`hp_tuning.py`**: Contains code for hyperparameter tuning using **Optuna**.
2. **`isothermal_base_model.py`**: Implements the isothermal base model, a predictive ML model trained on isothermal data. It does not use any temperature-property equations and takes temperature as a direct input.
3. **`multitemperature_base_model.py`**: Implements the multitemperature base model, trained on datasets covering five temperature levels. Like the isothermal model, it does not rely on equations.
4. **`eval_model.py`**: Implements the thermodynamics-informed model, including:
   - Converting SMILES to numerical descriptors using the **MORDRED Python package**.
   - Feature selection through a pipeline based on removing features with lowest variance, highest correlation with other features and highest number of missing values. Afterwards, ML based feature selection approach such as XGboost or Random Forest are available to select the most informative features with respect to the target values.
   - Training an ensemble of ANN models using **BAGGING** the training data.
   - Performing uncertainty assessment.
