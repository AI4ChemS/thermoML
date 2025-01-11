import pandas as pd
import numpy as np
import pubchempy as pcp
import pandas as pd
import requests
import rdkit, rdkit.Chem, rdkit.Chem.Draw
import mordred, mordred.descriptors








# def smile_extractor(df, descriptor_generating = True):

#     def chemical_name_to_smile(chemical_name, output_format='smiles'):
#         resolver_url = f'https://cactus.nci.nih.gov/chemical/structure/{chemical_name}/{output_format}'
        
#         try:
#             response = requests.get(resolver_url)
#             if response.status_code == 200:
#                 return response.text
#             else:
#                 return f"Error: Unable to resolve the chemical name. Status code: {response.status_code}"
#         except Exception as e:
#             return f"Error: {e}"
            
#     calc = mordred.Calculator(mordred.descriptors, ignore_3D = True)
#     for index,chem in df.iterrows():
#         df.loc[index,'smiles'] = chemical_name_to_smile(chem['Compounds'], 'smiles')
#         if descriptor_generating:
#             try:
#                 molecule = [rdkit.Chem.MolFromSmiles(chem['smiles'])]
#                 df.loc[index, calc.pandas(molecule).columns] = calc.pandas(molecule).values

#             except:
#                 print (f" Mordred descriptors are not measurable for {chem['Compounds']}; so it got removed from list of compounds")  
#                  df = df.drop(index)
#     df = df.reset_index(drop=True)            
#     return df

#-------------------------------------------------------------------------------------------------------------------------

# def mordred_descrptor_generator():

# calc = mordred.Calculator(mordred.descriptors, ignore_3D = True)
# molecules = [rdkit.Chem.MolFromSmiles(chem) for chem in df_chemicals['smiles']]


#-------------------------------------------------------------------------------------------------------------------------

def replace_string_with_nan(value):
    if 'a' in str(value):
        return np.nan
    elif 'b' in str(value):
        return np.nan
    else:
        return value

#-------------------------------------------------------------------------------------------------------------------------
def data_stratification(df,df_arrhenius_coeff, column_name = 'Ln(A)', 
                        bins = [-np.inf,-11.49,-9.68,-7.87,-6.06,-4.25,-2.44,np.inf], 
                        labels = [7,6,5,4,3,2,1]):
    df.insert(list(df.columns).index('smiles'), 
              f'Median_{column_name}_Categories', 
              pd.cut(df_arrhenius_coeff[column_name], bins ,labels))
    return df
#-------------------------------------------------------------------------------------------------------------------------

def arrhenius_type_fluid_identifier(df_arrhenius_coeff, metric = 'R_squared' , threshold = 0.9):
    valid_compounds = df_arrhenius_coeff.dropna(subset=[metric])
    valid_compounds = valid_compounds[valid_compounds[metric] >= threshold]
    return valid_compounds
#-------------------------------------------------------------------------------------------------------------------------

def find_highly_correlated_pairs(correlation_matrix, threshold=0.9):
    pairs_to_remove = set()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i+1, correlation_matrix.shape[1]):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                pairs_to_remove.add(i)
                pairs_to_remove.add(j)
    return pairs_to_remove 

#-------------------------------------------------------------------------------------------------------------------------


def data_curation (dataset_path , threshold_missing_values = 0.3, is_data_stratification = True):
    df = pd.read_excel(dataset_path, sheet_name = 'Descriptor_Mordred_Output')
    df_arrhenius_coeff = pd.read_excel(dataset_path, sheet_name = 'Linear Regression for Arre. Equ')
    df_eta_T = pd.read_excel(dataset_path, sheet_name = 'eta vs T')
    df_eta_T[df_eta_T.columns[1:]] = df_eta_T[df_eta_T.columns[1:]].astype(float)

# datapoints curation

    # check to see that the material follow the arrhenius type equation
    valid_compounds = arrhenius_type_fluid_identifier(df_arrhenius_coeff)
    df = df[df['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)
    df_arrhenius_coeff = df_arrhenius_coeff[df_arrhenius_coeff['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)
    df_eta_T = df_eta_T[df_eta_T['Compounds'].isin(valid_compounds['Compounds'])].reset_index(drop = True)
    

    

# Column (feature) curation

    
    # Adding arrhenius equation coefficients for each  material
    df.insert(1,'Ln(A)', df_arrhenius_coeff['Ln(A)'])
    df.insert(2,'Ea/R', df_arrhenius_coeff['Ea/R'])

    if is_data_stratification == True:
        df = data_stratification(df,df_arrhenius_coeff, column_name = 'Ln(A)', 
                                 bins = [-np.inf,-11.49,-9.68,-7.87,-6.06,-4.25,-2.44,np.inf], 
                                 labels = [7,6,5,4,3,2,1]) #dafault values of bins and labels
    
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:]    #feature_pool
    
    df[feature_names] = df[feature_names].applymap(replace_string_with_nan)


    # Drop columns with than NAN items more threshold values
    df = df.dropna(axis=1, thresh=int((1 - threshold_missing_values) * len(df)))
    features_start_at = list(df.columns).index('smiles')+1
    feature_names = df.columns[features_start_at:] 
    df [feature_names] = df[feature_names].fillna(0)

    # Drop columns with only zero values
    zero_columns = df.columns[df.eq(0).all()]
    df = df.drop(columns=zero_columns)

    return df, df_arrhenius_coeff, df_eta_T


#-------------------------------------------------------------------------------------------------------------------------


# Temeprature range selection

def temperature_selection(df, df_eta_T, Lower_temprature_limit = 250 , Upper_temperature_limit = 550 ):
    filtered_df = df_eta_T[(df_eta_T['Temperature'] >= Lower_temprature_limit) & (df_eta_T['Temperature'] <= Upper_temperature_limit)]

    # Group by "compound" and filter out groups with less than 5 entries
    filtered_df = filtered_df.groupby('Compounds').filter(lambda x: len(x) >= 5)
    filtered_df = filtered_df.reset_index(drop=True)

    
    def sort_within_chemical(df_eta_T):
        # Create an ordered categorical that preserves the original order of chemicals
        df_eta_T['Compounds'] = pd.Categorical(df_eta_T['Compounds'], categories=df_eta_T['Compounds'].unique(), ordered         = True)
        
        # Sort by 'Compounds' first (to maintain the order) and then by 'Temperature' within each group
        sorted_df = df_eta_T.sort_values(by=['Compounds', 'Temperature'])
        return sorted_df

    filtered_df = sort_within_chemical(filtered_df)

    def select_quantiles(filtered_df):
        quantiles = [0, 0.25, 0.5, 0.75, 1]
        selected_points = filtered_df.groupby('Compounds').quantile(q=quantiles, interpolation='midpoint').reset_index()
        selected_points.drop('level_1', axis = 1, inplace = True)
        return selected_points
    
    filtered_df = select_quantiles(filtered_df)


    temperature_invers_df = filtered_df.set_index(['Compounds', filtered_df.groupby('Compounds').cumcount()+1])['Temperature_inverse'].unstack()

    viscosity_ln_df = filtered_df.set_index(['Compounds', filtered_df.groupby('Compounds').cumcount()+1])['Viscosity_ln'].unstack()

    viscosity_df = filtered_df.set_index(['Compounds', filtered_df.groupby('Compounds').cumcount()+1])['Dynamic Viscosity '].unstack()

    
    temperature_invers_df.columns = [f'Temperature_inverse_{i}' for i in temperature_invers_df.columns]
    viscosity_ln_df.columns = [f'Viscosity_ln_{i}' for i in viscosity_ln_df.columns]
    viscosity_df.columns = [f'Viscosity_{i}' for i in viscosity_df.columns]


    df = pd.merge(df, temperature_invers_df, on='Compounds')
    df_viscosity_ln = pd.merge(df['Compounds'], viscosity_ln_df, on='Compounds')
    df_viscosity = pd.merge(df['Compounds'], viscosity_df, on='Compounds')


    return df , df_viscosity_ln, df_viscosity



    #-------------------------------------------------------------------------------------------------------------------------






    

    

    

    
    


    
    

    

    
    
        