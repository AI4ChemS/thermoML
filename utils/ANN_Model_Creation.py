import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Multiply, Concatenate, Add, Lambda, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1
from utils.data_curator import replace_string_with_nan
import os
import pickle
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# general information about dataset enter the ANN model (the final format of dataset that enter the model to evaluate BP fluids)
def dataset_general_info(df, test_dataset_name = "BP_fluids", train_dataset_name = "Mahyar_dataset"):
    general_info = {}
    #number and name of featrues (all should be reported in dictionary)
    features_start_at = list(df.columns).index('smiles')+1
    descriptor_names = list(df.columns[features_start_at:-5])    #feature_pool
    general_info ['descriptor_names'] = descriptor_names
    general_info ['descriptor_count'] = len(descriptor_names)
    general_info['compound_count'] = df.shape[0]
    general_info['train_dataset_name'] = train_dataset_name
    general_info['test_dataset_name'] = test_dataset_name
    return general_info


def ANN_predictor_model_defining(df, model_name, helper_output = True): #any df even df_test
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] #includes temperature
    
# Input layer (-number_of_temperature is because temperature points are also included in dataset)
    input_layer = Input(shape=(descriptors.shape[1]-5,))
    normalized_input = BatchNormalization()(input_layer)

    initializer_1 = tf.keras.initializers.HeNormal()
# no Lasso regulizer
    hidden_layer_1 = Dense(300, use_bias=False, kernel_initializer=initializer_1)(normalized_input)
# lasso regulizer
# hidden_layer_1 = Dense(300, use_bias=False, kernel_initializer=initializer_1, kernel_regularizer=l1(0.01))(normalized_input)
# Hidden layers with batch normalization
    hidden_layer_1 = BatchNormalization()(hidden_layer_1)
    hidden_layer_1 = Activation('relu')(hidden_layer_1)

    initializer_2 = tf.keras.initializers.HeNormal()
    hidden_layer_2 = Dense(150, use_bias=False, kernel_initializer = initializer_2)(hidden_layer_1)
    hidden_layer_2 = BatchNormalization()(hidden_layer_2)
    hidden_layer_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden_layer_2)

    initializer_3 = tf.keras.initializers.HeNormal()
    initializer_4 = tf.keras.initializers.HeNormal()

    # Output for bias (ln(A))
    output_bias = Dense(1,use_bias=False, kernel_initializer = initializer_3, name='output_bias')(hidden_layer_2)
    # Output for coefficient of Arrhenius equation (Ea/R)
    output_coefficient = Dense(1,use_bias=False, activation='linear', kernel_initializer = initializer_4, name='output_coefficient')(hidden_layer_2)

    temperature_invers_input = Input(shape=(5,))    
    
    
    output_multiply = Lambda(lambda x: x[0] * x[1], name='output_multiply')([output_coefficient, temperature_invers_input])

    final_output = Lambda(lambda x: x[0] + x[1], name='final_output')([output_bias, output_multiply])



    if helper_output == True:
        model = Model(inputs= [input_layer, temperature_invers_input], outputs = [final_output , output_bias, output_coefficient])

    
    else:
        model = Model(inputs= [input_layer, temperature_invers_input], outputs = final_output)    

#     general_info = dataset_general_info(df)
#     model_name = f"{general_info['train_dataset_name']}_{general_info ['feature_count']}_"
    
    base_path = 'Results'  # Modify this to your base path
    model_folder_path = os.path.join(base_path, model_name)
    os.makedirs(model_folder_path, exist_ok=True)
    

    return model , model_folder_path

# model_name could be obtained from above function output

def ANN_model_fitting(model, df, df_viscosity_ln, model_name, model_folder_path, epochs = 1000, batch_size = 32, validation_split = 0.2, helper_output = True, loss_weights = [0.85 , 0.15, 0], learning_rate = 0.01, patience = 150): 
    
    file_df_path = os.path.join(model_folder_path, 'df.pkl')    
    if not os.path.exists(file_df_path):
        with open(file_df_path, 'wb') as file:
            pickle.dump(df, file)

    file_df_viscosity_ln_path = os.path.join(model_folder_path, 'df_viscosity_ln.pkl')    
    if not os.path.exists(file_df_viscosity_ln_path):
        with open(file_df_viscosity_ln_path, 'wb') as file:
            pickle.dump(df_viscosity_ln, file)
            
    general_info = dataset_general_info(df, test_dataset_name = "BP_fluids", train_dataset_name = "Mahyar_dataset")
    file_general_info_path = os.path.join(model_folder_path, 'general_info.pkl')    
    if not os.path.exists(file_general_info_path):
        with open(file_general_info_path, 'wb') as file:
            pickle.dump(general_info, file)
      
        
       
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] 
    data_label = df_viscosity_ln.iloc[:,1:]
    bias_output_label = df.loc[:,'Ln(A)']
    coefficient_output_label = df.loc[:,'Ea/R']

    descriptor_names = list(descriptors.columns[:-5])
    file_descriptors_names_path = os.path.join(model_folder_path, 'descriptors_names.pkl')    
    if not os.path.exists(file_descriptors_names_path):
        with open(file_descriptors_names_path, 'wb') as file:
            pickle.dump(descriptor_names, file)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =  f"{model_folder_path}/{model_name}.keras",
                                                     save_best_only=True,
                                                     verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    if helper_output:
        target_outputs = [data_label.values, bias_output_label.values, coefficient_output_label.values]  # Two output arrays
        model.compile(optimizer=optimizer, 
          loss=['mae','mae', 'mae'],
          loss_weights = loss_weights,
          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]) 
    else:
        target_outputs = data_label.values  # One output array
        model.compile(optimizer=optimizer, 
              loss='mae',
              metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]) 

    history = model.fit(
        [descriptors.iloc[:, :-5].values, descriptors.iloc[:, -5:].values],
        target_outputs,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split,
        callbacks = [cp_callback, earlystopping_callback],
        verbose=0
    )
    
# Initial df_test must have 5 temperature enteries (if not, add zero columns to both df_test and df_viscosity_test. They will be deleted after prediction)


# Descriptor names are in the general info. So depending which model you want to use for prediction, you can mention its general_info_path here to read necessary descriptors names.

def test_data_preparation(df_test,df_viscosity_ln_test, general_info_path):
    model_general_info_path = general_info_path + "/general_info.pkl"
    with open(model_general_info_path, 'rb') as file:
        descriptor_names = pickle.load(file)["descriptor_names"]
    test_compounds_name = df_test['Compounds']
    df_test = pd.concat([df_test[descriptor_names], df_test.iloc[:,-5:]], axis = 1)
    df_test = df_test.applymap(replace_string_with_nan)
    df_test.fillna(0, inplace=True)
    try:
        df_test['Lipinski'] = df_test['Lipinski'].astype('float64')
        
    except:
        df_test = df_test
        
    true_label_test = df_viscosity_ln_test.iloc[:,1:]
    
    return df_test , true_label_test, test_compounds_name

# while predicting, you can set nymber of temperature interest in your model

def test_data_prediction(model,df_test,df_viscosity_ln_test, general_info_path, number_of_temperature_interest = 3, helper_output = True):
    
    df_test , true_label_test, test_compounds_name = test_data_preparation(df_test,df_viscosity_ln_test, general_info_path)
    
    if helper_output:
        predicted_label_test = model.predict([df_test.iloc[:,:-5].values,df_test.iloc[:,-5:].values])[0]
        ln_A_pred = model.predict([df_test.iloc[:,:-5].values,df_test.iloc[:,-5:].values])[1]
        Ea_pred = model.predict([df_test.iloc[:,:-5].values,df_test.iloc[:,-5:].values])[2]
    else:
        predicted_label_test = model.predict([df_test.iloc[:,:-5].values,df_test.iloc[:,-5:].values])
        ln_A_pred = None
        
    
    true_label_test = true_label_test.iloc[:,:number_of_temperature_interest]
    predicted_label_test = pd.DataFrame(predicted_label_test[:, :number_of_temperature_interest], columns = true_label_test.columns)
    
    return true_label_test, predicted_label_test , ln_A_pred , Ea_pred


# Uncertainty Assessment
    


# Bootstrapping

def createTrainingSets(df, df_viscosity_ln, N_sets = 20, train_set_size = 342):
    # df is train set here
    features_start_at = list(df.columns).index('smiles')+1
    train_indices = list(range(df.shape[0]))
    training_samples = [np.random.choice(train_indices, size = train_set_size, replace=False) for _ in range(N_sets)] 

    df_sets, df_viscosity_ln_sets = [], []

    for i in training_samples:
        df_sets.append(df.loc[i])
        df_viscosity_ln_sets.append(df_viscosity_ln.loc[i])
#         y_train_2_sets.append(df.loc[i].iloc[:,1])
    
    return df_sets, df_viscosity_ln_sets #, y_train_2_sets
                                        
                         
                         
def ensemble_ANN_models_training(df, df_viscosity_ln, N_sets = 20, epochs = 1000, batch_size = 32, validation_split = 0.2, helper_output = True, loss_weights = [0.85 , 0.15, 0], learning_rate = 0.01, train_set_size = 342, patience = 150, project_name = 'Default'):
    
    df_sets, df_viscosity_ln_sets = createTrainingSets(df, df_viscosity_ln, N_sets, train_set_size)
    for i in range(N_sets):
        print("================================================")
        print("Model : {}".format(i))
        df_i, df_viscosity_ln_i = df_sets[i], df_viscosity_ln_sets[i]
        model_i, model_folder_path_i = ANN_predictor_model_defining(df_i, model_name = "uncertainty_{}".format(i),  helper_output = True)
        ANN_model_fitting(model_i, df_i, df_viscosity_ln_i, model_name = "uncertainty_{}".format(i), model_folder_path = model_folder_path_i, epochs = epochs, batch_size = batch_size, validation_split = validation_split, helper_output = helper_output, loss_weights = loss_weights, learning_rate = learning_rate, patience = patience) 

        if i == N_sets - 1:
            break 
            

# df_test and true_label_test and test_compounds_name are output of test_data_preparation function
def ensemble_ANN_models_testing(df_test,df_viscosity_ln_test, general_info_path,  N_sets = 20, number_of_temperature = 3, model_path = '/Users/mahi/Desktop/AI-ML/Step_1_BO/Code_Files/Basic_ANN_Model/Results',
                                helper_output = True):
    
    predicted_label_test_set = []
    ln_A_set, Ea_set = [], []
    for i in range(N_sets):
#         model_i, model_folder_path_i = ANN_predictor_model_defining(df, model_name = "uncertainty_{}".format(i),  helper_output = True)
#         model_i.load_weights("{}/uncertainty_{}.keras".format(model_folder_path_i, i))
        model_i = tf.keras.models.load_model("{}/uncertainty_{}/uncertainty_{}.keras".format(model_path,i,i), safe_mode=False)

        true_label_test, predicted_label_test, ln_A_pred, Ea_pred = test_data_prediction(model_i,df_test, df_viscosity_ln_test, general_info_path = general_info_path, number_of_temperature_interest = number_of_temperature, helper_output = helper_output)
        
        predicted_label_test_set.append(predicted_label_test)
        ln_A_set.append(ln_A_pred)
        Ea_set.append(Ea_pred)


    ensemble_models_predictions , mean_pred_dict , std_pred_dict, std_pred_ln_A = {} , {} , {} , {}
    for j in range(number_of_temperature):
        # Concatenating the first column from each DataFrame
        ensemble_models_predictions[j+1] = pd.concat([predicted_label_test_set[i].iloc[:, j] for i in range(N_sets)], axis=1)
        mean_pred_dict[j+1] = pd.concat([predicted_label_test_set[i].iloc[:, j] for i in range(N_sets)], axis=1).mean(axis = 1)
        std_pred_dict[j+1] = pd.concat([predicted_label_test_set[i].iloc[:, j] for i in range(N_sets)], axis=1).std(axis = 1)
        


    
    mean_pred_df = pd.DataFrame(mean_pred_dict)
    mean_pred_df.columns = true_label_test.columns
    std_pred_df = pd.DataFrame(std_pred_dict)
    std_pred_df.columns = true_label_test.columns
    
    ln_A_df = pd.DataFrame(columns=range(N_sets))
    Ea_df = pd.DataFrame(columns=range(N_sets))
    

    l = 0
    for item in ln_A_set:
        ln_A_df[l] = item.flatten()
        l += 1
    
    l = 0
    for item in Ea_set:
        Ea_df[l] = item.flatten()
        l += 1
        
    ln_A_std_df = ln_A_df.std(axis = 1)
        
    
    
# Saving the results       
    directory = os.path.join(model_path, "Prediction_Results")
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    true_label_test.to_pickle(os.path.join(directory, 'true_label_test.pkl'))
    mean_pred_df.to_pickle(os.path.join(directory, 'mean_pred_df.pkl'))
    std_pred_df.to_pickle(os.path.join(directory, 'std_pred_df.pkl'))
    
    with open(os.path.join(directory, 'predicted_label_test_set.pkl'), 'wb') as file:
        pickle.dump(predicted_label_test_set, file)
        
    with open(os.path.join(directory, 'ensemble_models_predictions.pkl'), 'wb') as file:
        pickle.dump(ensemble_models_predictions, file)
    
    
    test_compounds_name = df_test['Compounds']
    
    return true_label_test, predicted_label_test_set, mean_pred_df, std_pred_df, test_compounds_name, ln_A_std_df



def uncertainty_plot(true_label_test, mean_pred_df, std_pred_df, test_compounds_name,ln_A_std_df, parity_plot = True, histogram_chart = True,
                     number_of_temperature=3, std_dev_threshold=0.25, actual_viscosity = True, 
                     model_path = '/Users/mahi/Desktop/AI-ML/Step_1_BO/Code_Files/Basic_ANN_Model/Results', uncertainty_mode = 'viscosity', parity_accurate_region_threshold = 0.3):

    
    temperature_list = ['Lower Temperature Range', 'Medium Temperature Range', 'Higher Temperature Range']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    
    # Parity_Plotting
    if uncertainty_mode == 'viscosity':
        uncertainty_flag_df = std_pred_df.iloc[:,0].copy()
        uncertainty_flag_df[std_pred_df.iloc[:,0] > std_dev_threshold] = 'r'
        uncertainty_flag_df[std_pred_df.iloc[:,0] <= std_dev_threshold] = 'b'
        
    elif uncertainty_mode == 'ln_A':
        uncertainty_flag_df = ln_A_std_df.copy()
        uncertainty_flag_df[ln_A_std_df > std_dev_threshold] = 'r'
        uncertainty_flag_df[ln_A_std_df <= std_dev_threshold] = 'b'
        
    
    
    if parity_plot:

        if uncertainty_mode == 'viscosity':
            for i in range(number_of_temperature):
                fig, axes = plt.subplots(1, 2, figsize=(16, 12))
                fig.subplots_adjust(wspace=4.5)# Create a figure with three subplots

                # Blue points
#                 blue_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df.iloc[:, i] == 'b']
#                 blue_points_true = true_label_test.iloc[:,i][uncertainty_flag_df.iloc[:, i] == 'b']
                blue_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df == 'b']
                blue_points_true = true_label_test.iloc[:,i][uncertainty_flag_df == 'b']

                # Calculate min and max values for plotting the lines
                min_val = min(blue_points_pred.min(), blue_points_true.min())
                max_val = max(blue_points_pred.max(), blue_points_true.max())

                # Scatter plot for predictions vs actual
                axes[0].scatter(blue_points_pred, blue_points_true, color='#3399ff', edgecolor = '#0073e6', s = 130 , alpha=1, label='Certain Predictions')

                # Parity line
                axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')

                # Parity line +/- standard deviation threshold
#                 axes[0].plot([min_val, max_val], [min_val + std_dev_threshold, max_val + std_dev_threshold], 'g--', label='+/- Standard Deviation')
#                 axes[0].plot([min_val, max_val], [min_val - std_dev_threshold, max_val - std_dev_threshold], 'g--')

                axes[0].plot([min_val, max_val], [min_val + parity_accurate_region_threshold, max_val + parity_accurate_region_threshold], 'g--', label='+/- Accuracy Threshold')
                axes[0].plot([min_val, max_val], [min_val - parity_accurate_region_threshold, max_val - parity_accurate_region_threshold], 'g--')                
                
                
                
                # Set labels and titles
                axes[0].set_xlabel('Predicted Ln(Dynamic Viscosity)', fontsize = 35)
                axes[0].set_ylabel('Actual Ln(Dynamic Viscosity)', fontsize = 35)
                axes[0].tick_params(axis='x', labelsize=30)  # Set x-tick label font size
                axes[0].tick_params(axis='y', labelsize=30)
#                 axes[0].set_title(f'Comparison at Temperature {i+1}, threshold:{std_dev_threshold}')
#                 axes[0].legend(fontsize = 20, )


                # Red Points
#                 red_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df.iloc[:, i] == 'r']
#                 red_points_true = true_label_test.iloc[:,i][uncertainty_flag_df.iloc[:, i] == 'r'] 
                red_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df == 'r']
                red_points_true = true_label_test.iloc[:,i][uncertainty_flag_df == 'r']  
                                                             

                min_val = min(red_points_pred.min(), red_points_true.min())
                max_val = max(red_points_pred.max(), red_points_true.max())

                # Scatter plot for predictions vs actual
                axes[1].scatter(red_points_pred, red_points_true, color='#ff6666', edgecolor = '#cc0000', s = 130, alpha=1, label='Uncertain Predictions')

                # Parity line
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')

                # Parity line +/- standard deviation threshold
        #         axes[i,1].plot([min_val, max_val], [min_val + std_dev_threshold, max_val + std_dev_threshold], 'g--', label='+/- Standard Deviation')
        #         axes[i,1].plot([min_val, max_val], [min_val - std_dev_threshold, max_val - std_dev_threshold], 'g--')

        
        
        
                srcc, _ = spearmanr(blue_points_pred, blue_points_true)
                print(f"Spearman Rank Correlation Coefficient (SRCC): {srcc}")

                mae = mean_absolute_error(blue_points_true, blue_points_pred)
                print(f"Mean Absolute Error (MAE): {mae}")

                mse = mean_squared_error(blue_points_true, blue_points_pred)
                print(f"Mean Squared Error (MSE): {mse}")

                r2 = r2_score(blue_points_true, blue_points_pred)
                print(f"Coefficient of Determination (R2): {r2}")


            
                # Set labels and titles
                axes[1].set_xlabel('Predicted Ln(Dynamic Viscosity)', fontsize = 35)
                axes[1].set_ylabel('Actual Ln(Dynamic Viscosity)', fontsize = 35)
                axes[1].tick_params(axis='x', labelsize=30)  # Set x-tick label font size
                axes[1].tick_params(axis='y', labelsize=30)
#                 axes[1].set_title(f'Comparison at Temperature {i+1}, threshold:{std_dev_threshold}')
                

#                 axes[1].legend( fontsize = 20)


                blue_line = mlines.Line2D([], [], color='#3399ff', marker='o', markersize=20, label='Certain Predictions', linestyle='None')
                red_line = mlines.Line2D([], [], color='#ff6666', marker='o', markersize=20, label='Uncertain Predictions', linestyle='None')
                parity_line = mlines.Line2D([], [], color='red', linestyle='--', label='Perfect Fit')

                # Place a single legend at the bottom center of the figure
                if i ==2:
                    fig.legend(handles=[blue_line, red_line, parity_line], loc='lower center', ncol=3, fontsize=40, 
                           bbox_to_anchor=(0.5, 0.01))

                # Adjust layout to make room for the bottom legend
                fig.suptitle(f'{temperature_list[i]}', fontsize=45, verticalalignment='top')


                plt.tight_layout(rect=[0, 0.11, 1, 1])
                directory = os.path.join(model_path, "Prediction_Results", "Figures")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f'{directory}/temperature_parity_{i+1}.svg', bbox_inches='tight') 
                plt.show()

###################################################################################################################
                
                
        if uncertainty_mode == 'ln_A':
            for i in range(number_of_temperature):
                fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # Create a figure with three subplots

                # Blue points
                blue_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df == 'b']
                blue_points_true = true_label_test.iloc[:,i][uncertainty_flag_df == 'b']

                # Calculate min and max values for plotting the lines
                min_val = min(blue_points_pred.min(), blue_points_true.min())
                max_val = max(blue_points_pred.max(), blue_points_true.max())

                # Scatter plot for predictions vs actual
                axes[0].scatter(blue_points_pred, blue_points_true, color='#3399ff', edgecolor = '#0073e6' , alpha=0.6, label='Predicted vs Actual')

                # Parity line
                axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')

                # Parity line +/- standard deviation threshold
                axes[0].plot([min_val, max_val], [min_val + parity_accurate_region_threshold, max_val + parity_accurate_region_threshold], 'g--', label='+/- Standard Deviation')
                axes[0].plot([min_val, max_val], [min_val - parity_accurate_region_threshold, max_val - parity_accurate_region_threshold], 'g--')

                # Set labels and titles
                axes[0].set_xlabel('Predicted Ln(Dynamic Viscosity)')
                axes[0].set_ylabel('Actual Ln(Dynamic Viscosity)')
#                 axes[0].set_title(f'Comparison at Temperature {i+1}, threshold:{std_dev_threshold}, parity_accurate_region_threshold:{parity_accurate_region_threshold}')
                axes[0].legend()


                # Red Points
                red_points_pred = mean_pred_df.iloc[:,i][uncertainty_flag_df == 'r']
                red_points_true = true_label_test.iloc[:,i][uncertainty_flag_df == 'r']        

                min_val = min(red_points_pred.min(), red_points_true.min())
                max_val = max(red_points_pred.max(), red_points_true.max())

                # Scatter plot for predictions vs actual
                axes[1].scatter(red_points_pred, red_points_true, color='#ff6666', edgecolor = '#cc0000', alpha=0.9, label='Predicted vs Actual')

                # Parity line
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')

                # Parity line +/- standard deviation threshold
        #         axes[i,1].plot([min_val, max_val], [min_val + std_dev_threshold, max_val + std_dev_threshold], 'g--', label='+/- Standard Deviation')
        #         axes[i,1].plot([min_val, max_val], [min_val - std_dev_threshold, max_val - std_dev_threshold], 'g--')

                # Set labels and titles
                axes[1].set_xlabel('Predicted Ln(Dynamic Viscosity)')
                axes[1].set_ylabel('Actual Ln(Dynamic Viscosity)')
#                 axes[1].set_title(f'Comparison at Temperature {i+1}, threshold:{std_dev_threshold}')
                axes[1].legend()

                plt.tight_layout()
                directory = os.path.join(model_path, "Prediction_Results", "Figures")
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(f'{directory}/temperature_parity_{i+1}.svg') 
                plt.show()

                
                
                
                

    def autolabel(ax, rects, rotation=90):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=rotation, fontsize=12)
    # Histogram BP

    if histogram_chart:
        bar_width = 0.35
        
        for i in range(number_of_temperature):
            fig, ax = plt.subplots(figsize=(15, 12))  # Create a new figure for each temperature

            sorted_indices = true_label_test.iloc[:, i].sort_values().index
            sorted_predictions = mean_pred_df.iloc[sorted_indices, i]
            sorted_true_values = true_label_test.iloc[sorted_indices, i]
            sorted_compound_names = test_compounds_name[sorted_indices]
            certainty_colors = uncertainty_flag_df.iloc[sorted_indices, i].map({'b': 'blue', 'r': 'red'})

            index = np.arange(len(sorted_compound_names))

            if actual_viscosity:
                rects1 = ax.bar(index - bar_width/2, np.exp(sorted_predictions), bar_width, label='Predicted', color='blue')
                rects2 = ax.bar(index + bar_width/2, np.exp(sorted_true_values), bar_width, label='Actual', color='orange')
                ax.set_yscale('log')
                ax.set_ylabel('Dynamic Viscosity (cp)')
                

            else:
                rects1 = ax.bar(index - bar_width/2, sorted_predictions, bar_width, label='Predicted', color='blue')
                rects2 = ax.bar(index + bar_width/2, sorted_true_values, bar_width, label='Actual', color='orange')
                ax.set_ylabel('ln(Dynamic Viscosity)')

            autolabel(ax, rects1)
            autolabel(ax, rects2)
            ax.set_xticks(index)
            ax.set_xticklabels(sorted_compound_names, rotation=90, ha="right")

            # Color the x-axis labels based on certainty
            for ticklabel, tickcolor in zip(ax.get_xticklabels(), certainty_colors):
                ticklabel.set_color(tickcolor)

            ax.set_title(f'Temperature {i+1}')
            ax.legend()
            plt.tight_layout()
            directory = os.path.join(model_path, "Prediction_Results", "Figures")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f'{directory}/temperature_histogram_{i}.svg', bbox_inches='tight') 
            plt.show() 
            

    return uncertainty_flag_df




import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def tuned_ANN_predictor_model_defining(df, model_name, helper_output=True):
    features_start_at = list(df.columns).index('smiles') + 1
    feature_names = df.columns[features_start_at:]  # feature_pool
    descriptors = df[feature_names]  # includes temperature
    
    # Input layer
    input_layer = Input(shape=(descriptors.shape[1]-5,))
    normalized_input = BatchNormalization()(input_layer)
    
    # Dynamically adding layers based on optimized hyperparameters
    x = normalized_input
    # Layer definitions based on optimization results
    x = Dense(256, activation='selu', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(324, activation='elu', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(327, activation='elu', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(460, activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dense(134, activation='selu', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Outputs remain unchanged
    output_bias = Dense(1, use_bias=False, kernel_initializer='he_normal', name='output_bias')(x)
    output_coefficient = Dense(1, use_bias=False, activation='linear', kernel_initializer='he_normal', name='output_coefficient')(x)

    temperature_invers_input = Input(shape=(5,))
    output_multiply = Lambda(lambda x: x[0] * x[1], name='output_multiply')([output_coefficient, temperature_invers_input])
    final_output = Lambda(lambda x: x[0] + x[1], name='final_output')([output_bias, output_multiply])

    if helper_output:
        model = Model(inputs=[input_layer, temperature_invers_input], outputs=[final_output, output_bias])
    else:
        model = Model(inputs=[input_layer, temperature_invers_input], outputs=final_output)    

    base_path = '/Users/mahi/Desktop/AI-ML/Step_1_BO/Code_Files/Basic_ANN_Model/Results'
    model_folder_path = os.path.join(base_path, model_name)
    os.makedirs(model_folder_path, exist_ok=True)
    

    return model, model_folder_path

          

    
    
def ANN_Base_Model(df, model_name):  # any df even df_test
    features_start_at = list(df.columns).index('smiles') + 1
    feature_names = df.columns[features_start_at:]  # feature_pool
    descriptors = df[feature_names]  # includes temperature

    # Input layer
    input_layer = Input(shape=(descriptors.shape[1]-4,))
    normalized_input = BatchNormalization()(input_layer)

    # Initializer for the first hidden layer
    initializer_1 = tf.keras.initializers.HeNormal()

    # First hidden layer without Lasso regularizer
    hidden_layer_1 = Dense(300, use_bias=False, kernel_initializer=initializer_1)(normalized_input)

    # Batch normalization and activation
    hidden_layer_1 = BatchNormalization()(hidden_layer_1)
    hidden_layer_1 = Activation('relu')(hidden_layer_1)

    # Initializer for the second hidden layer
    initializer_2 = tf.keras.initializers.HeNormal()
    hidden_layer_2 = Dense(150, use_bias=False, kernel_initializer=initializer_2)(hidden_layer_1)
    hidden_layer_2 = BatchNormalization()(hidden_layer_2)
    hidden_layer_2 = tf.keras.layers.LeakyReLU(alpha=0.1)(hidden_layer_2)

    # Initializer for additional layers (if any)
    initializer_3 = tf.keras.initializers.HeNormal()
    initializer_4 = tf.keras.initializers.HeNormal()

    # Final layer for regression
    output_layer = Dense(1, activation='linear')(hidden_layer_2)

    # Creating the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    base_path = 'Results'  # Modify this to your base path
    model_folder_path = os.path.join(base_path, model_name)
    os.makedirs(model_folder_path, exist_ok=True)
    

    return model , model_folder_path

def ANN_Base_Bodel_fitting(model, df, df_viscosity_ln, model_name, model_folder_path, epochs = 1000, batch_size = 32, validation_split = 0.2, learning_rate = 0.01, patience = 150): 
    
        
       
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    descriptors = df[feature_names] 
    data_label = df_viscosity_ln.iloc[:,1:]



    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =  f"{model_folder_path}/{model_name}.keras",
                                                     save_best_only=True,
                                                     verbose=1)
    earlystopping_callback = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)


    target_outputs = data_label.values  # One output array
    model.compile(optimizer=optimizer, 
          loss='mae',
          metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()]) 

    history = model.fit(
        descriptors.iloc[:, :-4].values,
        descriptors.iloc[:,0].values,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split,
        callbacks = [cp_callback, earlystopping_callback],
        verbose=0
    )









