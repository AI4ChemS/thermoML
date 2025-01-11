import optuna
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Lambda

def tune_hyperparameters(X_train, y_train, ln_A_train, Ea_R_train):
    def objective(trial):
        print('-------------------------')
        print('begining of the tuning section with Optuna')
        # Hyperparameter suggestions
        num_layers = trial.suggest_int('num_layers', 2, 6)  # Number of hidden layers
        neurons_per_layer = trial.suggest_int('neurons_per_layer', 50, 300, step=50)  # Neurons per layer
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])

        # Define the model
        input_layer = Input(shape=(X_train.shape[1] - 5,))
        x = BatchNormalization()(input_layer)

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

        model = Model(inputs=[input_layer, temp_inv], outputs=[mu_pred, ln_A_pred, Ea_R_pred])

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=['mae', 'mae', 'mae'],
            loss_weights=[0.8, 0.2, 0],
            metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
        temp_list = ['temp_inv_1', 'temp_inv_2', 'temp_inv_3', 'temp_inv_4', 'temp_inv_5']
        history = model.fit(
            [X_train.drop(columns=temp_list).values, X_train[temp_list].values],
            [y_train.values, ln_A_train.values, Ea_R_train.values],
            epochs=300,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Return the validation loss
        return history.history['val_loss'][-1]

    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    return study.best_params
