from findhr.preprocess.example_mappings import MatchBinary, MatchOrdinal, MatchFeatureInclusion, MatchFeatureSet
from findhr.preprocess.mapping import AttachMetadata, DetachMetadata, DerivedColumn
from sklearn.pipeline import Pipeline
from metadata import md_CDS_JDS_ADS
from MACRO import MacroVariables
import pandas as pd

from utils import convert_cols


def load_dataset(fair_data=True):
    # Read dataset
    df_JDS = pd.read_csv(MacroVariables.FILEPATH_JOB_OFFERS,  # converters for columns of lists of values
                         converters={c:convert_cols for c in ["Age_j", "Competences_j", "Knowledge_j", "Languages_j"]})
    df_CDS = pd.read_csv(MacroVariables.FILEPATH_CURRICULA,  # converters for columns of lists of values
                         converters={c:convert_cols for c in ["Age_c", "Experience_c", "Competences_c", "Knowledge_c", "Languages_c"]})
    df_ADS_FAIR = pd.read_csv(MacroVariables.FILEPATH_ADS_FAIR)
    df_ADS_UNFAIR = pd.read_csv(MacroVariables.FILEPATH_ADS_UNFAIR)

    # Define subsets of columns
    cols_id = ['qId', 'kId']
    # Define the subset of columns of the HUDD dataset describing the candidate,
    # which are used in the preprocessing+prediction pipeline
    cols_c = ['Education_c', 'Age_c', 'Gender_c', 'Contract_c',
              'Nationality_c', 'Competences_c', 'Knowledge_c', 'Languages_c',
              'Experience_c']
    cols_j = ['Education_j', 'Age_j', 'Gender_j',  'Contract_j', 'Nationality_j', 'Competences_j',
              'Knowledge_j', 'Languages_j', 'Experience_j']
    cols_pred_preprocess = cols_c + cols_j
    cols_not_for_pred = ['Occupation_c', 'Occupation_j']
    cols_sensitive = ['Gender_c']
    col_target = ['score']

    if fair_data:
        # Merge CDS and JDS through ADS in a single dataframe
        df_CDS_JDS = pd.merge(df_ADS_FAIR, df_JDS, on='qId')
    else:
        # Merge CDS and JDS through ADS in a single dataframe
        df_CDS_JDS = pd.merge(df_ADS_UNFAIR, df_JDS, on='qId')

    df_CDS_JDS = pd.merge(df_CDS, df_CDS_JDS, on='kId')
    df_CDS_JDS = df_CDS_JDS[cols_id + [col for col in df_CDS_JDS if col not in cols_id+col_target] + col_target ]

    return df_CDS_JDS, {'cols_pred_preprocess': cols_pred_preprocess,
                        'cols_sensitive': cols_sensitive,
                        'cols_id': cols_id,
                        'cols_not_for_pred': cols_not_for_pred,
                        'col_target': col_target}

def build_matching_functions():
    # Matching functions for pairs of job-candidate features
    maps_matching = {
         # MatchBinary: 1 = job value = candidate value OR job value is 'Any' OR candidate value is 'Any', 0 = otherwise
        # (('qId',), ('qId',)): IdentityMapping(),
        # (('kId',), ('kId',)): IdentityMapping(),
        # (('rank',), ('rank',)): IdentityMapping(),
        (('Contract_j', 'Contract_c'), ('fitness_Contract',)): MatchBinary(),
        (('Gender_j', 'Gender_c'), ('fitness_Gender',)): MatchBinary(),
        (('Nationality_j', 'Nationality_c'), ('fitness_Nationality',)): MatchBinary(),

         # MatchOrdinal: 1 = job value >= candidate OR job value is 'Any', 0 = otherwise
        (('Education_j', 'Education_c'), ('fitness_Education',)): MatchOrdinal(),
        (('Experience_j', 'Experience_c'), ('fitness_Experience',)): MatchOrdinal(),

         # MatchFeatureInclusion: 1 = candidate value in (job value(0,), >= job value(1,)) OR job value is 'Any', 0 = otherwise
        (('Age_j', 'Age_c'), ('fitness_Age',)): MatchFeatureInclusion(),

         # MatchFeatureSet: 1 = fraction of job value that appear in candidate value
        (('Languages_j', 'Languages_c'), ('fitness_Languages',)): MatchFeatureSet(),
        (('Competences_j', 'Competences_c'), ('fitness_Competences',)): MatchFeatureSet(),
        (('Knowledge_j', 'Knowledge_c'), ('fitness_Knowledge',)): MatchFeatureSet()
    }
    return maps_matching


def build_fitness_matrix(df_CDS_JDS, cols_dict, fair_data=True):
    maps_matching = build_matching_functions()

    # Calculation as fit-transform preprocessing
    pipeline_fitness = Pipeline(steps=[
        ("init", AttachMetadata(md_CDS_JDS_ADS)),
        ("matching", DerivedColumn(maps_matching)),
        ("end", DetachMetadata())
    ])

    pipeline_fitness.fit(X=df_CDS_JDS)
    fitness_matrix = pipeline_fitness.transform(X=df_CDS_JDS)
    df_fitness_mat = fitness_matrix.copy(deep=True)
    columns_keep = cols_dict['cols_id'] + \
                   [col for col in fitness_matrix if
                    col.startswith('fitness_')] + cols_dict['cols_sensitive'] + cols_dict['col_target']

    df_fitness_mat = df_fitness_mat[columns_keep]

    # From scores, we can learn regressors; or we can produce ranks, and learn ranking models
    df_fitness_mat['rank'] = df_fitness_mat.groupby("qId")['score'].rank('dense', ascending=False)
    df_fitness_mat['rank'] = df_fitness_mat['rank'].apply(lambda x: x if x <= MacroVariables.TOP_K else MacroVariables.TOP_K + 1)

    return pipeline_fitness, df_fitness_mat
