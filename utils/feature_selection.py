import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import xgboost as xgb





# Features always come after SMILES string columm




def removing_correlated_features_operator(df):  
    
    def find_highly_correlated_features(correlation_matrix, threshold = 0.9):
        pairs_to_remove = set()
        for i in range(correlation_matrix.shape[0]):
            for j in range(i+1, correlation_matrix.shape[1]):
                if abs(correlation_matrix.iloc[i, j]) >= threshold:
                    # pairs_to_remove.add(i)
                    pairs_to_remove.add(j)
        return pairs_to_remove

    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names]
    
    while True:
        correlation_matrix =  descriptors.corr()
        pairs_to_remove = find_highly_correlated_features(correlation_matrix)
        
        # Break the loop if no more highly correlated pairs are found
        if len(pairs_to_remove) == 0:
            break
        
        # Remove descriptors involved in highly correlated pairs
        descriptors =  descriptors.drop(descriptors.columns[list(pairs_to_remove)], axis=1)

    df = pd.concat([df.iloc[:,:features_start_at],descriptors], axis = 1)
    return df


def variance_threshold_operator(df, threshold = 0.005):
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names]
    
    pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('variance_threshold', VarianceThreshold(threshold)),  ])
    descriptors_high_variance = pipeline.fit_transform(descriptors)
    # Get the names of the selected features
    selected_feature_indices = pipeline.named_steps['variance_threshold'].get_support(indices=True)
    selected_feature_names = descriptors.columns[selected_feature_indices]
    df = pd.concat([df.iloc[:,:features_start_at],df[selected_feature_names]], axis = 1)

    return df




def LASSO_feature_selection_operator (df, df_viscosity_ln, alpha_lower_limit = -2, alpha_upper_limit = 0):
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] 
    if 'Temperature_inverse_1' in df.columns:
        descriptors = descriptors.iloc[:,:-4] 
    data_label = df_viscosity_ln['Viscosity_ln_1']
        
    scaler = StandardScaler()
    descriptors_standardized = scaler.fit_transform(descriptors)

    lasso = LassoCV(alphas=np.logspace(alpha_lower_limit, alpha_upper_limit, 100), cv=5).fit(descriptors_standardized, data_label)
    selected_features = descriptors.columns[lasso.coef_ != 0]
    coef = lasso.coef_
    selected_features = descriptors.columns[coef != 0]
    
    if 'Temperature_inverse_1' in df.columns:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features], df.iloc[:,-4:]],axis = 1)
    else:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features]],axis = 1)
        
    return df



def RF_feature_selection_operator(df, df_viscosity_ln, top_n_percent_features = 80):
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] 
    if 'Temperature_inverse_1' in df.columns:
        descriptors = descriptors.iloc[:,:-4] 
    data_label = df_viscosity_ln['Viscosity_ln_1']
    rf = RandomForestRegressor()
    rf.fit(descriptors, data_label)
    importances = rf.feature_importances_
    selected_features = descriptors.columns[importances > np.percentile(importances, top_n_percent_features)]  # top 20% features

    if 'Temperature_inverse_1' in df.columns:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features], df.iloc[:,-4:]],axis = 1)
    else:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features]],axis = 1)
        
    return df


def XGBoost_feature_selection_operator(df, df_viscosity_ln, importance_threshold = 30):
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] 
    if 'Temperature_inverse_1' in df.columns:
        descriptors = descriptors.iloc[:,:-4] 
    data_label = df_viscosity_ln['Viscosity_ln_1']
    
    model = XGBRegressor(
    objective='reg:squarederror',  # Objective function for regression
    n_estimators=100,             # Number of trees
    max_depth=3,                  # Depth of each tree
    learning_rate=0.1,            # Learning rate
    subsample=0.8,                # Subsample ratio of the training instances
    colsample_bytree=0.8          # Subsample ratio of columns when constructing each tree
    )


    model.fit(descriptors.values, data_label.values)
    importances = model.feature_importances_
    
    # Sort the feature importances and select the k-th last element as the threshold
    threshold = np.sort(model.feature_importances_)[-1*importance_threshold]
    
    selection = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = selection.transform(descriptors)
    
    selected_mask = selection.get_support() 
    selected_features = descriptors.columns[selected_mask] 

    if 'Temperature_inverse_1' in df.columns:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features], df.iloc[:,-4:]],axis = 1)
    else:
        df = pd.concat([df.iloc[:,:features_start_at], df[selected_features]],axis = 1)
        
    return df
    


    






    





    
        
