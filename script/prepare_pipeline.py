# For now, reuse the code from the previous notebooks for the FINDHR course
#%%
import pandas as pd
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from findhr.preprocess.metadata import JSONMetadata, validate_schema
# Disable warning for SettingWithCopyWarning
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'

# Path and datasets
PATH = pathlib.Path("../../course_findhr/for_students/data_notebooks/data")

SUFFIX_DATASET = '1' # '1' for demonstration, '2' for practice

FILENAME_CURRICULA = f"curricula{SUFFIX_DATASET}.csv"
FILENAME_JOB_OFFERS = f"job_offers{SUFFIX_DATASET}.csv"
FILENAME_ADS_FAIR = f'score{SUFFIX_DATASET}_fair.csv'
FILENAME_ADS_UNFAIR = f'score{SUFFIX_DATASET}_unfair.csv'

FILENAME_FITNESS_MATRIX_FAIR = f"fitness_mat{SUFFIX_DATASET}_fair.csv"
FILENAME_FITNESS_MATRIX_UNFAIR = f"fitness_mat{SUFFIX_DATASET}_unfair.csv"

# Define the top k for the ranking, i.e., the number of candidates to be shortlisted for each job offer
TOP_K = 10 # max 31 for LightGBM Ranker

import ast
# Utility function to convert raw data in CSV
def convert(x):
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, list):
        return x
    try:
        x = ast.literal_eval(x)
    finally:
        return x


# Read dataset
df_JDS = pd.read_csv(PATH / FILENAME_JOB_OFFERS, # converters for columns of lists of values
            converters={c:convert for c in ["Age_j", "Competences_j", "Knowledge_j", "Languages_j"]})
df_CDS = pd.read_csv(PATH/ FILENAME_CURRICULA, # converters for columns of lists of values
            converters={c:convert for c in ["Age_c", "Experience_c", "Competences_c", "Knowledge_c", "Languages_c"]})
df_ADS_FAIR = pd.read_csv(PATH/ FILENAME_ADS_FAIR)
df_ADS_UNFAIR = pd.read_csv(PATH/ FILENAME_ADS_UNFAIR)

# Define the metadata for the JDS dataset
md_JDS = {
    'qId': JSONMetadata(schema={'type': 'number'}),
    'Occupation_j': JSONMetadata(schema={'type': 'string'}),
    'Education_j': JSONMetadata(schema={'enum': ['No education', 'Degree', 'Bachelor D.', 'Master D.', 'PhD', 'Any']},
                              attr_type='category'),
    'Age_j': JSONMetadata(schema={'type': 'array',
                                  'prefixItems': [
                                    { 'type': 'number' },
                                    { 'type': 'number' },
                                  ],
                                  'items': False},
                          ),
    'Gender_j': JSONMetadata(schema={'enum': ['Male', 'Female', 'Non-binary', 'Any']},
                             attr_type='category', attr_usage='sensitive'),
    'Contract_j': JSONMetadata(schema={'enum': ['Remote', 'Hybrid', 'In presence']}),
    'Nationality_j': JSONMetadata(schema={'type': 'string'}),
    'Competences_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Knowledge_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'} }),
    'Languages_j': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Experience_j': JSONMetadata(schema={'type': 'number'}),
}

# Define the metadata for the CDS dataset
md_CDS = {
    'kId': JSONMetadata(schema={'type': 'integer'}),
    'Occupation_c': JSONMetadata(schema={'type': 'string'}),
    'Education_c': JSONMetadata(schema={'enum': ['No education', 'Degree', 'Bachelor D.', 'Master D.', 'PhD', 'Any']},
                              attr_type='category'),
    'Age_c': JSONMetadata(schema={'type': 'number'}),
    'Gender_c': JSONMetadata(schema={'enum': ['Male', 'Female', 'Non-binary']},
                             attr_type='category', attr_usage='sensitive'),
    'Contract_c': JSONMetadata(schema={'enum': ['Remote', 'Hybrid', 'In presence', 'Any']}, attr_type='category'),
    'Nationality_c': JSONMetadata(schema={'type': 'string'}),
    'Competences_c': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Knowledge_c': JSONMetadata(schema={'type': "array", 'items': {'type': 'string'}}),
    'Experience_c': JSONMetadata(schema={'type': 'number'}),
    'Languages_c': JSONMetadata(schema={'type': "array",'items': {'type': 'string'}}),
}


# Validate JSON schema for the CDS and JDS datasets
if False: # this cell may take 40-80 seconds to run
    print('{} Validating schema for df_CDS {}'.format('-' * 20, '-' * 20))
    validate_schema(df_CDS, md_CDS)

    print('{} Validating schema for df_JDS {}'.format('-' * 20, '-' * 20))
    validate_schema(df_JDS, md_JDS)


# Join the metadata of the CDS and JDS datasets
md_CDS_JDS = {**md_CDS, **md_JDS}


# Define subsets of columns
cols_id = ['qId', 'kId']
cols_pred = ['Education_c', 'Age_c', 'Gender_c', 'Contract_c',
       'Nationality_c', 'Competences_c', 'Knowledge_c', 'Languages_c',
       'Experience_c', 'Education_j', 'Age_j', 'Gender_j',
       'Contract_j', 'Nationality_j', 'Competences_j',
       'Knowledge_j', 'Languages_j', 'Experience_j']
cols_not_for_pred = ['Occupation_c', 'Occupation_j']
cols_sensitive = ['Gender_c']
col_target = ['score']

# Merge CDS and JDS through ADS in a single dataframe
df_CDS_JDS_FAIR = pd.merge(df_ADS_FAIR, df_JDS, on='qId')
df_CDS_JDS_FAIR = pd.merge(df_CDS, df_CDS_JDS_FAIR, on='kId')
df_CDS_JDS_FAIR = df_CDS_JDS_FAIR[ cols_id + [col for col in df_CDS_JDS_FAIR if col not in cols_id+col_target] + col_target ]
df_CDS_JDS_FAIR.head()