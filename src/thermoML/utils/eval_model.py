import pandas as pd
import numpy as np
import pubchempy as pcp
import pandas as pd
import requests
import warnings
import os
import pickle
import rdkit, rdkit.Chem, rdkit.Chem.Draw
import mordred, mordred.descriptors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import rdkit, rdkit.Chem, rdkit.Chem.Draw
import mordred, mordred.descriptors
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Multiply, Concatenate, Add, Lambda, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1
from .hp_tuning import tune_hyperparameters
#-------------------------------------------------------------------------------------------------------------------------



def generate_2d_features(df):
    if 'smiles' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'smiles' column.")

        
        
    calc = mordred.Calculator(mordred.descriptors, ignore_3D = True)
    molecules = [rdkit.Chem.MolFromSmiles(chem) for chem in df['smiles']]
    features = calc.pandas(molecules)
    result_df = pd.concat([df, features], axis=1)

    return result_df



def arrhenius_fluid_identifier(df_arrhenius_coeff, metric = 'R_squared' , threshold = 0.9):
    valid_compounds = df_arrhenius_coeff.dropna(subset=[metric])
    valid_compounds = valid_compounds[valid_compounds[metric] >= threshold]
    return valid_compounds


def stratifier(df,arr_data, column_name = 'Ln_A', 
                        bins = [-np.inf,-11.49,-9.68,-7.87,-6.06,-4.25,-2.44,np.inf], 
                        labels = [7,6,5,4,3,2,1]):
    df.insert(list(df.columns).index('smiles'), 
              f'Median_{column_name}_Categories', 
              pd.cut(arr_data[column_name], bins ,labels))
    return df

def replace_string_with_nan(value):
    if 'a' in str(value):
        return np.nan
    elif 'b' in str(value):
        return np.nan
    else:
        return value


def find_highly_correlated_features(correlation_matrix, corr_threshold):
    pairs_to_remove = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            if abs(correlation_matrix.iloc[i, j]) >= corr_threshold:
                pairs_to_remove.add(j)
    return pairs_to_remove
    
def sort_temp_within_chemical(mu_temp):
    mu_temp['Compounds'] = pd.Categorical(mu_temp['Compounds'], categories=mu_temp['Compounds'].unique(), ordered = True)
    mu_temp_sorted = mu_temp.sort_values(by=['Compounds', 'temp'])
    return mu_temp_sorted
    
def select_quantiles(mu_temp):
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    selected_points = mu_temp.groupby('Compounds').quantile(q=quantiles, interpolation='midpoint').reset_index()
    selected_points.drop('level_1', axis = 1, inplace = True)
    return selected_points  


def ANN_model_training(X_train, y_train, ln_A_train, Ea_R_train, model_path, helper_output = True, patience = 150, learning_rate = 0.01, loss_weights = [0.85 , 0.15, 0], epochs = 1000):

    input_layer = Input(shape=(X_train.shape[1]-5,))
    normalized_input = BatchNormalization()(input_layer)

    initializer_1 = tf.keras.initializers.HeNormal()
    hidden_layer_1 = Dense(300, use_bias=False, kernel_initializer=initializer_1)(normalized_input)
    hidden_layer_1 = BatchNormalization()(hidden_layer_1)
    hidden_layer_1 = Activation('relu')(hidden_layer_1)

    initializer_2 = tf.keras.initializers.HeNormal()
    hidden_layer_2 = Dense(150, use_bias=False, kernel_initializer = initializer_2)(hidden_layer_1)
    hidden_layer_2 = BatchNormalization()(hidden_layer_2)
    hidden_layer_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden_layer_2)

    initializer_3 = tf.keras.initializers.HeNormal()
    initializer_4 = tf.keras.initializers.HeNormal()

    ln_A_pred = Dense(1,use_bias=False, kernel_initializer = initializer_3, name='ln_A_pred')(hidden_layer_2)
    Ea_R_pred = Dense(1,use_bias=False, activation='linear', kernel_initializer = initializer_4, name='Ea_R_pred')(hidden_layer_2)

    temp_inv = Input(shape=(5,))   
    Ea_RT = Lambda(lambda x: x[0] * x[1], name='Ea_RT')([Ea_R_pred, temp_inv])

    mu_pred = Lambda(lambda x: x[0] + x[1], name='mu')([ln_A_pred, Ea_RT])

    if helper_output == True:
        model = Model(inputs= [input_layer, temp_inv], outputs = [mu_pred , ln_A_pred, Ea_R_pred])

    else:
        model = Model(inputs= [input_layer, temp_inv], outputs = mu_pred)        
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =  f"{model_path}/model.keras",
                                                     save_best_only=True,
                                                     verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    if helper_output:
        y_true = [y_train.values, ln_A_train.values, Ea_R_train.values] 
        model.compile(optimizer=optimizer, 
          loss=['mae','mae', 'mae'],
          loss_weights = loss_weights,
          #metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]) 
          metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])
    else:
        y_true = y_train.values  
        model.compile(optimizer=optimizer, 
              loss='mae',
              metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])  
              #metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])   
    
    temp_list = ['temp_inv_1', 'temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
    history = model.fit(
        [X_train.drop(columns = temp_list).values, X_train[temp_list].values],
        y_true,
        epochs = epochs,
        batch_size = 32,
        validation_split = 0.2,
        callbacks = [cp_callback, earlystopping_callback],
        verbose=0
    )
    
    return model




def ANN_model_training_tuned(X_train, y_train, ln_A_train, Ea_R_train, model_path, 
                       num_layers, neurons_per_layer, learning_rate, activation,
                       helper_output=True, loss_weights=[0.85, 0.15, 0], epochs=1000):

    input_layer = Input(shape=(X_train.shape[1] - 5,))
    normalized_input = BatchNormalization()(input_layer)

    x = normalized_input
    for _ in range(num_layers):
        initializer = tf.keras.initializers.HeNormal()
        x = Dense(neurons_per_layer, use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        if activation == 'relu':
            x = Activation('relu')(x)
        else:
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    ln_A_pred = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.HeNormal(), name='ln_A_pred')(x)
    Ea_R_pred = Dense(1, use_bias=False, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal(), name='Ea_R_pred')(x)
    temp_inv = Input(shape=(5,))
    Ea_RT = Lambda(lambda x: x[0] * x[1], name='Ea_RT')([Ea_R_pred, temp_inv])
    mu_pred = Lambda(lambda x: x[0] + x[1], name='mu')([ln_A_pred, Ea_RT])


    if helper_output == True:
        model = Model(inputs= [input_layer, temp_inv], outputs = [mu_pred , ln_A_pred, Ea_R_pred])

    else:
        model = Model(inputs= [input_layer, temp_inv], outputs = mu_pred)
        
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =  f"{model_path}/model.keras",
                                                     save_best_only=True,
                                                     verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(patience = 200, restore_best_weights = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
    if helper_output:
        y_true = [y_train.values, ln_A_train.values, Ea_R_train.values] 
        model.compile(optimizer=optimizer, 
          loss=['mae','mae', 'mae'],
          loss_weights = loss_weights,
          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]) 
    else:
        y_true = y_train.values  
        model.compile(optimizer=optimizer, 
              loss='mae',
              metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])     
    
    temp_list = ['temp_inv_1', 'temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
    history = model.fit(
        [X_train.drop(columns = temp_list).values, X_train[temp_list].values],
        y_true,
        epochs = epochs,
        batch_size = 32,
        validation_split = 0.2,
        callbacks = [cp_callback, earlystopping_callback],
        verbose=0
    )
    
    return model


    
def main_training(df, arr_data, mu_temp, path, to_drop = ['Compounds', 'smiles'], is_data_stratification = False, lim_nan_values = 0.3, corr_filter = True, corr_threshold = 0.9, var_filter = True, var_threshold = 0.02, feature_selection = 'xgboost', n_models = 20, set_size = 342, helper_output = True):
    
    df = generate_2d_features(df)

    mu_temp[list(set(mu_temp.columns) - set(to_drop))] = mu_temp[list(set(mu_temp.columns) - set(to_drop))].astype(float)

    valid_compounds = arrhenius_fluid_identifier(arr_data)
    valid_compounds = arrhenius_fluid_identifier(arr_data)
    df = df[df['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)
    arr_data = arr_data[arr_data['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)
    mu_temp = mu_temp[mu_temp['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)

    df.insert(1,'Ln_A', arr_data['Ln_A'])
    df.insert(2,'Ea_R', arr_data['Ea_R'])    

    if is_data_stratification == True:
        str_column = 'Ln_A'
        df = stratifier(df,arr_data, column_name = str_column, 
                                 bins = [-np.inf,-11.49,-9.68,-7.87,-6.06,-4.25,-2.44,np.inf], 
                                 labels = [7,6,5,4,3,2,1])  
        to_drop.append('Median_{column_name}_Categories')


    to_drop.extend(['Ln_A', 'Ea_R'])
    df[list(set(df.columns) - set(to_drop))] = df[list(set(df.columns) - set(to_drop))].applymap(replace_string_with_nan)
    df = df.dropna(axis=1, thresh=int((1 - lim_nan_values) * len(df)))
    df  = df.fillna(0)
    zero_columns = df.columns[df.eq(0).all()]
    df = df.drop(columns=zero_columns)
    
    if corr_filter == True:
        features = df[list(set(df.columns) - set(to_drop))]
        while True:
            correlation_matrix =  features.corr()
            pairs_to_remove = find_highly_correlated_features(correlation_matrix, corr_threshold = 0.9)

            if len(pairs_to_remove) == 0:
                break

            features =  features.drop(features.columns[list(pairs_to_remove)], axis=1).reset_index(drop=True)
        df = pd.concat([df[to_drop], features], axis = 1)

    if var_filter == True:
        features = df[list(set(df.columns) - set(to_drop))]
        pipeline = Pipeline([
                             ('scaler', MinMaxScaler()),
                             ('variance_threshold', VarianceThreshold(var_threshold))
                            ])        
        pipeline.fit_transform(features)
        high_var_features_idx = pipeline.named_steps['variance_threshold'].get_support(indices=True)
        high_var_features = features.columns[high_var_features_idx]
        df = pd.concat([df[to_drop], features[high_var_features]], axis = 1)
        
    print('--------------------------------------')
    print('df initial curation ended!')
################################################################################################################
    warnings.filterwarnings('ignore')
    lower_temp_lim = 250
    upper_temp_lim = 550
    mu_temp = mu_temp[(mu_temp['temp'] >= lower_temp_lim) & (mu_temp['temp'] <= upper_temp_lim)]
    mu_temp = mu_temp.groupby('Compounds').filter(lambda x: len(x) >= 5)
    mu_temp = mu_temp.reset_index(drop=True)
    mu_temp = sort_temp_within_chemical(mu_temp)
    mu_temp = select_quantiles(mu_temp)
    
    temp_inv = mu_temp.set_index(['Compounds', mu_temp.groupby('Compounds').cumcount()+1])['temp_inv'].unstack()
    temp_inv.columns = [f'temp_inv_{i}' for i in temp_inv.columns]
    
    mu_log = mu_temp.set_index(['Compounds', mu_temp.groupby('Compounds').cumcount()+1])['mu_log'].unstack()
    mu_log.columns = [f'mu_log_{i}' for i in mu_log.columns]
    
    mu = mu_temp.set_index(['Compounds', mu_temp.groupby('Compounds').cumcount()+1])['mu'].unstack()
    mu.columns = [f'mu_{i}' for i in mu.columns]
    
    df = pd.merge(df, temp_inv, on='Compounds')
    df_mu_log = pd.merge(df['Compounds'], mu_log, on='Compounds')
    df_mu = pd.merge(df['Compounds'], mu, on='Compounds')

    print('--------------------------------------')
    print('temperature adding ended!')
################################################################################################################

    if feature_selection == 'lasso':
        alpha_lower_limit = -1.5
        alpha_upper_limit = 0

        temp_to_drop = ['temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
        features = df[list(set(df.columns) - set(to_drop) - set(temp_to_drop))]
        label = df_mu_log['mu_log_1']
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)
        lasso = LassoCV(alphas=np.logspace(alpha_lower_limit, alpha_upper_limit, 100), cv=5).fit(features_std, label)
        lasso_coef = lasso.coef_
        lasso_features = list(features.columns[lasso_coef != 0])
        lasso_features.remove('temp_inv_1')
        df = pd.concat([df[to_drop], features[lasso_features], features['temp_inv_1'], df[temp_to_drop]], axis = 1)
        
    if feature_selection == 'rf':
        rf_threshold_percent = 80 
        temp_to_drop = ['temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
        features = df[list(set(df.columns) - set(to_drop) - set(temp_to_drop))]
        label = df_mu_log['mu_log_1']
        rf = RandomForestRegressor()
        rf.fit(features, label)
        rf_coef = rf.feature_importances_
        rf_features = list(features.columns[rf_coef > np.percentile(rf_coef, rf_threshold_percent)])
        rf_features.remove('temp_inv_1')        
        df = pd.concat([df[to_drop], features[rf_features], features['temp_inv_1'], df[temp_to_drop]], axis = 1)


    if feature_selection == 'xgboost':
        xgboost_threshold = 30
        temp_to_drop = ['temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
        features = df[list(set(df.columns) - set(to_drop) - set(temp_to_drop))]
        label = df_mu_log['mu_log_1']        
        
        model = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,           
        max_depth=3,                 
        learning_rate=0.1,            
        subsample=0.8,               
        colsample_bytree=0.8          
        )
        
        model.fit(features.values, label.values)
        xgboost_coef = model.feature_importances_
        selection_threshold = np.sort(xgboost_coef)[-1*xgboost_threshold]
        selection = SelectFromModel(model, threshold=selection_threshold, prefit=True)
        selection.transform(features)
        xgboost_features = list(features.columns[selection.get_support()])
        xgboost_features.remove('temp_inv_1')        
        df = pd.concat([df[to_drop], features[xgboost_features], features['temp_inv_1'], df[temp_to_drop]], axis = 1)

    temp_to_drop = ['temp_inv_1', 'temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
    features_list = list(df[list(set(df.columns) - set(to_drop) - set(temp_to_drop))].columns)    
    features_path = os.path.join(path, 'features.pkl') 
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(features_path):
        with open(features_path, 'wb') as file:
            pickle.dump(features_list, file)
            
    print('--------------------------------------')
    print('ML feature selection ended!')
################################################################################################################

    idx_range = list(range(df.shape[0]))
    idx_sets = [np.random.choice(idx_range, size = set_size, replace=False) for _ in range(n_models)] 

    df_sets, df_mu_log_sets = [], []
    for i in idx_sets:
        df_sets.append(df.loc[i])
        df_mu_log_sets.append(df_mu_log.loc[i])

    for i in range(n_models):
        print("================================================")
        print("Model : {}".format(i))
        df_train, df_mu_log_train = df_sets[i], df_mu_log_sets[i]
        X_train = df_train[list(set(df_train.columns) - set(to_drop))]
        y_train = df_mu_log_train.drop(columns=to_drop, errors='ignore')
        Ln_A_train = df_train.loc[:,'Ln_A']
        Ea_R_train = df_train.loc[:,'Ea_R']

        model_path = os.path.join(path, str(i))
        os.makedirs(model_path, exist_ok=True)
        model = ANN_model_training(X_train, y_train, Ln_A_train, Ea_R_train, model_path, helper_output = True, patience = 250, learning_rate = 0.01, loss_weights = [0.8 , 0.2, 0], epochs = 1000) 
        
    return df, df_mu_log, df_mu
 #-------------------------------------------------------------------------------------------------------------------------
   
def main_test(df_test, temp_test, path, helper_output = True, n_models = 20, to_drop = ['Compounds', 'smiles']):
    with open(os.path.join(path, 'features.pkl'), 'rb') as file:
        features_list = pickle.load(file)
    
    df_test = generate_2d_features(df_test)
    features = df_test[list(set(df_test.columns) - set(to_drop))]
    features = features.applymap(replace_string_with_nan)
    features  = features.fillna(0)
    features = features[features_list]
    temp_test = 1/temp_test
    
    n_temp = temp_test.shape[1]
    if n_temp < 5:
        n_zero_pad = 5 - n_temp        
        for i in range(n_temp + 1, n_temp + n_zero_pad + 1):
            temp_test[f'temp_inv_{i}'] = 0
    
    y_pred_set = []
    ln_A_pred_set, Ea_R_pred_set = [], []                      
    for i in range(n_models):
        model = tf.keras.models.load_model("{}/{}/model.keras".format(path,i), safe_mode=False)
        
        if helper_output:
            y_pred = model.predict([features.values, temp_test.values])[0]
            Ln_A_pred = model.predict([features.values, temp_test.values])[1]
            Ea_R_pred = model.predict([features.values, temp_test.values])[2]
        else:
            y_pred = model.predict([features.values, temp_test.values])
            ln_A_pred = None
        
        y_col_name = ['mu_log_1', 'mu_log_2', 'mu_log_3', 'mu_log_4', 'mu_log_5']
        y_pred = pd.DataFrame(y_pred[:, :n_temp], columns = y_col_name[:n_temp])        
        y_pred_set.append(y_pred)
        ln_A_pred_set.append(Ln_A_pred)
        Ea_R_pred_set.append(Ea_R_pred)
        
    y_pred_df , y_pred_avg , y_pred_std, std_pred_ln_A = {} , {} , {} , {}
    for j in range(n_temp):
        y_pred_df[j+1] = pd.concat([y_pred_set[i].iloc[:, j] for i in range(n_models)], axis=1)
        y_pred_avg[j+1] = pd.concat([y_pred_set[i].iloc[:, j] for i in range(n_models)], axis=1).mean(axis = 1)
        y_pred_std[j+1] = pd.concat([y_pred_set[i].iloc[:, j] for i in range(n_models)], axis=1).std(axis = 1)
        
        
    y_pred_avg = pd.DataFrame(y_pred_avg)
    y_pred_avg.columns = y_col_name[:n_temp]
    y_pred_std = pd.DataFrame(y_pred_std)
    y_pred_std.columns = ['std_mu_log_1', 'std_mu_log_2', 'std_mu_log_3', 'std_mu_log_4', 'std_mu_log_5']  
    
    
    ln_A_pred_df = pd.DataFrame(columns=range(n_models))
    Ea_R_pred_df = pd.DataFrame(columns=range(n_models))
    
    l = 0
    for item in ln_A_pred_set:
        ln_A_pred_df[l] = item.flatten()
        l += 1
    ln_A_pred_avg = ln_A_pred_df.mean(axis = 1)
    ln_A_pred_std = ln_A_pred_df.std(axis = 1)
    
    l = 0
    for item in Ea_R_pred_set:
        Ea_R_pred_df[l] = item.flatten()
        l += 1
    Ea_R_pred_avg = Ea_R_pred_df.mean(axis = 1)
    Ea_R_pred_std = Ea_R_pred_df.std(axis = 1)
    
    output = pd.concat([df_test[to_drop], temp_test.iloc[:,:n_temp], y_pred_avg, y_pred_std], axis = 1)
    output['Ln_A_avg'] = ln_A_pred_avg
    output['Ln_A_std'] = ln_A_pred_std
    output['Ea_R_avg'] = Ea_R_pred_avg
    output['Ea_R_std'] = Ea_R_pred_std
    
    
    return output
        


        

    

    
    