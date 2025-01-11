import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Multiply, Concatenate, Add, Lambda, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1
from data_curator import replace_string_with_nan
import os
import pickle




def ANN_classifier_model_defining(df, model_name, helper_output = True):
#     features_start_at = list(df.columns).index('smiles')+1
#     feature_names = df.columns[features_start_at:]  # feature_pool
#     descriptors = df[feature_names]

    # Input layer (-5 is because temperature points are also included in dataset)
    input_layer = Input(shape=(df.shape[1],))
    normalized_input = BatchNormalization()(input_layer)

    initializer_1 = tf.keras.initializers.HeNormal()
    hidden_layer_1 = Dense(300, use_bias=False, kernel_initializer=initializer_1)(normalized_input)
    hidden_layer_1 = BatchNormalization()(hidden_layer_1)
    hidden_layer_1 = Activation('relu')(hidden_layer_1)

    initializer_2 = tf.keras.initializers.HeNormal()
    hidden_layer_2 = Dense(150, use_bias=False, kernel_initializer=initializer_2)(hidden_layer_1)
    hidden_layer_2 = BatchNormalization()(hidden_layer_2)
    hidden_layer_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden_layer_2)

    # Binary classification output layer
    initializer_3 = tf.keras.initializers.HeNormal()
    output_layer = Dense(1, use_bias=True, activation='sigmoid', kernel_initializer=initializer_3)(hidden_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model  # Assuming model_folder_path is not needed or should be set elsewhere




def ANN_classifier_fitting(model, X_train, y_train, model_name, epochs = 1000, batch_size = 32, validation_split = 0.2, helper_output = True, loss_weights = [0.85 , 0.15], learning_rate = 0.01): 
    

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =  f"LLM/{model_name}.keras",
                                                     save_best_only=True,
                                                     verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(patience = 150, restore_best_weights = True)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train.values,
        y_train.values,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split,
        callbacks = [cp_callback, earlystopping_callback],
        verbose=0
    )



    
def createTrainingSets(df, df_viscosity_ln, train_indices, N_sets = 20, train_set_size = 50):
    # df is train set here
    training_samples = [np.random.choice(train_indices, size = train_set_size, replace=False) for _ in range(N_sets)] 

    df_sets, df_viscosity_ln_sets = [], []

    for i in training_samples:
        df_sets.append(df.loc[i])
        df_viscosity_ln_sets.append(df_viscosity_ln.loc[i])
#         y_train_2_sets.append(df.loc[i].iloc[:,1])
    
    return df_sets, df_viscosity_ln_sets #, y_train_2_sets
                                        
                         
                         
def ensemble_ANN_models_training(df, df_viscosity_ln, train_set_size = 50, N_sets = 20, epochs = 1000, batch_size = 32, validation_split = 0.2, helper_output = True, loss_weights = [0.85 , 0.15], learning_rate = 0.01):
    
    df_sets, df_viscosity_ln_sets = createTrainingSets(df, df_viscosity_ln, N_sets, train_set_size)
    for i in range(N_sets):
        print("================================================")
        print("Model : {}".format(i))
        df_i, df_viscosity_ln_i = df_sets[i], df_viscosity_ln_sets[i]
        model_i = ANN_classifier_model_defining(df_i, model_name = "uncertainty_{}".format(i),  helper_output = True)
        ANN_classifier_fitting(model_i, df_i, df_viscosity_ln_i, train_set_size = train_set_size, model_name = "uncertainty_{}".format(i), epochs = epochs, batch_size = batch_size, validation_split = validation_split, helper_output = helper_output, loss_weights = loss_weights, learning_rate = learning_rate) 

        if i == N_sets - 1:
            break 

