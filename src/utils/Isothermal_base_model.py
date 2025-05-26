import os
import pandas as pd
import numpy as np
import optuna
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.thermoML import replace_string_with_nan
from utils.thermoML import generate_2d_features


def clean_array(array):
    array = np.array(array, dtype='object')    
    cleaned_array = np.zeros_like(array, dtype='float64')
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            try:
                cleaned_array[i, j] = float(value)
            except (ValueError, TypeError):
                cleaned_array[i, j] = 0.0  
    return cleaned_array


def create_model(trial, X_train):
    units_1 = trial.suggest_int("units_1", 100, 300, step=50)
    units_2 = trial.suggest_int("units_2", 50, 200, step=50)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "LeakyReLU"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    
    input_layer = Input(shape=(X_train.shape[1],))
    normalized_input = BatchNormalization()(input_layer)

    hidden_layer_1 = Dense(units_1, use_bias=False, kernel_initializer="he_normal")(normalized_input)
    hidden_layer_1 = BatchNormalization()(hidden_layer_1)
    if activation == "LeakyReLU":
        hidden_layer_1 = LeakyReLU(alpha=0.1)(hidden_layer_1)
    else:
        hidden_layer_1 = Activation(activation)(hidden_layer_1)

    hidden_layer_2 = Dense(units_2, use_bias=False, kernel_initializer="he_normal")(hidden_layer_1)
    hidden_layer_2 = BatchNormalization()(hidden_layer_2)
    if activation == "LeakyReLU":
        hidden_layer_2 = LeakyReLU(alpha=0.1)(hidden_layer_2)
    else:
        hidden_layer_2 = Activation(activation)(hidden_layer_2)

    output_layer = Dense(1, use_bias=False, name="mu_pred")(hidden_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mae", metrics=["mse", "root_mean_squared_error"])


    return model

def objective(trial):
    model = create_model(trial)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[
            cp_callback, tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) 
        ],
    )

    val_loss = min(history.history["val_loss"])
    return val_loss


def main_base_train_ST(df, df_mu, set_size, base_model_path = "Results/base_model_singleT", n_models = 20, hp_tuning = False):
    
    temp_columns = ['temp_inv_1', 'temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
    temp = 'temp_inv_3'
    df_base = df.drop(columns = temp_columns)
    df_base['temperature'] = df[temp].values
    df_base['temperature'] = 1/df_base['temperature']

    mu_columns = ['mu_1', 'mu_2', 'mu_3', 'mu_4', 'mu_5']
    mu = 'mu_3'
    df_mu_base = df_mu.drop(columns = mu_columns)
    df_mu_base['viscosity'] = df_mu[mu].values    

    if hp_tuning == True:
        study = optuna.create_study(direction="minimize")  
        study.optimize(objective, n_trials=50)  
    else:
        with open(base_model_path + "/best_trial.pkl", "rb") as f:
            best_trial = pickle.load(f)
            
    idx_range = list(range(df_base.shape[0]))
    idx_sets = [np.random.choice(idx_range, size = set_size, replace=False) for _ in range(n_models)]
    df_sets, df_mu_sets = [], []
    for i in idx_sets:
        df_sets.append(df_base.loc[i])
        df_mu_sets.append(df_mu_base.loc[i])
        
    train_loss_list, val_loss_list = [], []
    for j in range(n_models):
        print("================================================")
        print("Model : {}".format(j))
        df_train, df_mu_train = df_sets[j], df_mu_sets[j]
        X_train = df_train.iloc[:,4:].values
        y_train = df_mu_train['viscosity'].values

        best_model = create_model(best_trial, X_train)
        history  = best_model.fit(X_train, y_train, validation_split=0.2, epochs=400, batch_size=32,
                                 callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath =  base_model_path +
                                                                                 f"/{j}/model.keras",
                                                                                 save_best_only=True,verbose=1), 
                                  tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)])
        train_loss = min(history.history['loss']) 
        val_loss = min(history.history['val_loss'])
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)   
    
    with open(base_model_path + "/features_list.pkl", "wb") as f:
        pickle.dump(df_train.iloc[:,4:-1].columns, f)
    
    
    
def main_base_test_ST(df_test, temp_test, base_model_path, threshold = 5, to_drop = ['Compounds', 'smiles']):
   
    with open(os.path.join(base_model_path, 'features_list.pkl'), 'rb') as file:
        features_list = pickle.load(file)
    
    df_test = generate_2d_features(df_test)
    features = df_test[list(set(df_test.columns) - set(to_drop))]
    features = features.applymap(replace_string_with_nan)
    features  = features.fillna(0)
    features = features[features_list]
    temp_list = ['T1','T2','T3','T4','T5']
    df_feat_T = pd.concat([features, temp_test[temp_list]], axis = 1)
    
    df_test_melted = df_feat_T.melt(id_vars=[col for col in df_feat_T.columns if col not in temp_list], 
                                    value_vars=temp_list,
                                    var_name='temperature_label', 
                                    value_name='temperature')
    
    df_test_melted = df_test_melted.drop(columns=['temperature_label'])
    X_test = df_test_melted.values
    X_test = clean_array(X_test)
    
    y_pred_base_list = []
    for i in range(20):
        model = tf.keras.models.load_model("{}/{}/model.keras".format(base_model_path, i), safe_mode=False)
        y_pred_base_list.append(np.squeeze(model.predict(X_test)))
        
    y_pred_base_df = pd.DataFrame(y_pred_base_list).T
    y_pred_base_df = np.log(y_pred_base_df)
    y_pred_base_df_avg = y_pred_base_df.mean(axis = 1)
    y_pred_base_df_std = y_pred_base_df.std(axis = 1)
    y_pred_base_certainty = y_pred_base_df_std < threshold

    y_pred_base_avg_temp =  pd.DataFrame(y_pred_base_df_avg.values.reshape(5, int(df_test_melted.shape[0]/5)).T)
    y_pred_base_std_temp =  pd.DataFrame(y_pred_base_df_std.values.reshape(5, int(df_test_melted.shape[0]/5)).T)
    y_pred_base_certainty_temp =  pd.DataFrame(y_pred_base_certainty.values.reshape(5, int(df_test_melted.shape[0]/5)).T)
    
    return y_pred_base_avg_temp, y_pred_base_std_temp, y_pred_base_certainty_temp


