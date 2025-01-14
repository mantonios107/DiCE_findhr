import argparse
import dice_ml
from MACRO import MacroVariables
from ranking import ranking_pipeline
from preprocessing import load_dataset, build_fitness_matrix
from super_model import SuperRankerPipeline
from sklearn.model_selection import train_test_split
from utils import rank2relevance
from lightgbm import LGBMRanker
import pandas as pd
from utils import convert_cols_mod
from preprocessing import build_matching_functions
from findhr.preprocess.mapping import AttachMetadata, DetachMetadata, DerivedColumn
from sklearn.pipeline import Pipeline
from metadata import md_CDS_JDS_ADS
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', '-j', type=int, default=161,  # 160-199
                        help='The job id for which the counterfactual explanation is to be generated')

    parser.add_argument('--candidate_position', '-c', type=int, default=14,  # 16
                        help='The position of the candidate in the ranking for which the counterfactual explanation is to be generated')

    parser.add_argument('--explanation_method', '-m', type=str, choices=['random', 'genetic', 'kdtree'],
                        default='random', help='The method for generating counterfactual explanations')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--target_rank', '-r', type=int, default=MacroVariables.TOP_K,
                       help='The target rank for the counterfactual explanation')
    group.add_argument('--target_score', '-s', type=float)

    return parser.parse_args()


def data_split(df_qId_kId):
    all_jobs = df_qId_kId['qId'].unique()
    train_jobs, test_jobs = train_test_split(all_jobs, test_size=0.2, random_state=42, shuffle=False)
    train_jobs, val_jobs = train_test_split(train_jobs, test_size=0.25, random_state=42, shuffle=False)

    # Build train, test and validation sets, ensuring they are sorted by qId, kId
    df_train = df_qId_kId[df_qId_kId['qId'].isin(train_jobs)].sort_values(["qId", "kId"])
    df_val = df_qId_kId[df_qId_kId['qId'].isin(val_jobs)].sort_values(["qId", "kId"])
    df_test = df_qId_kId[df_qId_kId['qId'].isin(test_jobs)].sort_values(["qId", "kId"])

    return df_train, df_val, df_test

def load_dataset(fair_data=True):
    # Read dataset
    df_JDS = pd.read_csv(MacroVariables.FILEPATH_JOB_OFFERS,  # converters for columns of lists of values
                         converters={c:convert_cols_mod for c in ["Age_j", "Competences_j", "Knowledge_j", "Languages_j"]})
    df_CDS = pd.read_csv(MacroVariables.FILEPATH_CURRICULA,  # converters for columns of lists of values
                         converters={c:convert_cols_mod for c in ["Age_c", "Experience_c", "Competences_c", "Knowledge_c", "Languages_c"]})

    df_ADS_FAIR = pd.read_csv(MacroVariables.FILEPATH_ADS_FAIR)
    df_ADS_UNFAIR = pd.read_csv(MacroVariables.FILEPATH_ADS_UNFAIR)


    cols_dict_HUDD = define_cols_dict_HUDD()

    if fair_data:
        # Merge CDS and JDS through ADS in a single dataframe
        df_CDS_JDS = pd.merge(df_ADS_FAIR, df_JDS, on='qId')
    else:
        # Merge CDS and JDS through ADS in a single dataframe
        df_CDS_JDS = pd.merge(df_ADS_UNFAIR, df_JDS, on='qId')

    df_CDS_JDS = pd.merge(df_CDS, df_CDS_JDS, on='kId')
    df_CDS_JDS = df_CDS_JDS[cols_dict_HUDD['cols_id'] + [col for col in df_CDS_JDS if col not in cols_dict_HUDD['cols_id']+ cols_dict_HUDD['col_target']] +
                            cols_dict_HUDD['col_target']]
    df_CDS_JDS[cols_dict_HUDD['col_rank']] = np.minimum(df_CDS_JDS.groupby("qId")[cols_dict_HUDD['col_target']].rank('dense', ascending=False), MacroVariables.TOP_K + 1)

    # TODO: Change the original data
    return df_CDS_JDS, cols_dict_HUDD


def define_cols_dict_HUDD():

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
    col_rank = ['rank']

    # Define the subset of columns of the HUDD dataset for the counterfactual explanation
    outcome_name_col = 'lambda'  # 'pred_rank'
    continuous_features = ['Age_c', 'Experience_c', 'Experience_j']  # ['Age_c', 'Experience_c'],
    categorical_features = ['Education_c', 'Gender_c',
       'Contract_c', 'Nationality_c', 'Competences_c', 'Knowledge_c',
       'Languages_c', 'Education_j',
       'Gender_j', 'Contract_j', 'Nationality_j', 'Competences_j',
       'Knowledge_j', 'Languages_j', 'Age_j']
    cols_pred = ['Education_c', 'Age_c', 'Gender_c',
       'Contract_c', 'Nationality_c', 'Competences_c', 'Knowledge_c',
       'Languages_c', 'Experience_c', 'Education_j', 'Age_j',
       'Gender_j', 'Contract_j', 'Nationality_j', 'Competences_j',
       'Knowledge_j', 'Languages_j', 'Experience_j']
    # continuous_features + categorical_features
    return {'outcome_name_col': outcome_name_col, 'continuous_features': continuous_features,
            'categorical_features': categorical_features, 'cols_pred': cols_pred,
            'cols_id': cols_id, 'cols_sensitive': cols_sensitive, 'col_target': col_target, 'col_rank': col_rank,
            'cols_pred_preprocess': cols_pred_preprocess, 'cols_not_for_pred': cols_not_for_pred}

def define_cols_dict_FEDD():

    outcome_name_col = 'lambda'  # 'pred_rank'
    continuous_features = ['fitness_Languages', 'fitness_Competences',
                           'fitness_Knowledge']  # ['Age_c', 'Experience_c'],
    categorical_features = ['fitness_Contract', 'fitness_Nationality', 'fitness_Education', 'fitness_Experience',
                            'fitness_Age', 'fitness_Gender']
    cols_pred = continuous_features + categorical_features

    cols_id = ['qId', 'kId']  # ids
    cols_sensitive = ['Gender_c']  # sensitive attribute(s)
    col_target = 'score'  # target value for ranking
    col_rank = 'rank'  # rank value for ranking

    return {'outcome_name_col': outcome_name_col, 'continuous_features': continuous_features,
            'categorical_features': categorical_features, 'cols_pred': cols_pred,
            'cols_id': cols_id, 'cols_sensitive': cols_sensitive, 'col_target': col_target, 'col_rank': col_rank}

def extract_explicand_data_cf(job_id, exp_c_pred_rank, pipeline_fitness, df_CDS_JDS):
    # df_qId contains the data for the job qId

    # Isolate the candidates' profiles applying for the job qId
    df_qId_HUDD = df_CDS_JDS[df_CDS_JDS['qId'] == job_id]

    # Extract the explicand candidate kId
    exp_c_kId = df_qId_HUDD.loc[df_qId_HUDD['pred_rank'] == exp_c_pred_rank, 'kId'].iloc[0]

    # Isolate the explicand candidate profile
    exp_c_profile = df_CDS_JDS[df_CDS_JDS['kId'] == exp_c_kId]

    exp_c_fitness = pipeline_fitness.transform(exp_c_profile)

    exp_c = {'kId': exp_c_kId, 'profile': exp_c_profile, 'fitness': exp_c_fitness}

    return df_qId_HUDD, exp_c


def prepare_data_cf(df_qId_HUDD, cols_dict):
    # Convert data types
    df_qId_HUDD_pre = df_qId_HUDD[cols_dict['cols_pred']].copy(deep=True) #.astype('int').copy(deep=True)
    df_qId_HUDD_pre[cols_dict['cols_id']] = df_qId_HUDD[cols_dict['cols_id']]
    #df_qId_HUDD_pre[cols_dict['categorical_features']].copy(deep=True) #.astype('int').copy(deep=True)
    # df_qId_HUDD_pre[cols_dict['continuous_features']] = df_qId_HUDD[cols_dict['continuous_features']].astype(
    #    'float').copy(deep=True)
    df_qId_HUDD_pre[cols_dict['outcome_name_col']] = df_qId_HUDD[cols_dict['outcome_name_col']].copy(deep=True)
    feature_dtypes = None # {col: df_qId_HUDD_pre[col].dtype for col in df_qId_HUDD_pre[cols_dict['cols_pred']].columns}

    return df_qId_HUDD_pre, feature_dtypes


def define_target(args, df_qId_HUDD):
    # 'in_top_k' or 'out_top_k' depending on the candidate position
    explicand_class = 'in_top_k' if args.candidate_position <= MacroVariables.TOP_K else 'out_top_k'

    # target rank for counterfactual explanation
    if args.target_rank:
        tgt_cf_rank = args.target_rank
        tgt_cf_score = df_qId_HUDD[df_qId_HUDD['pred_rank'] == tgt_cf_rank]['score'].iloc[0]
        tgt_cf_candidate = df_qId_HUDD[df_qId_HUDD['pred_rank'] == tgt_cf_rank]

    elif args.target_score:
        tgt_cf_rank = None
        tgt_cf_score = args.target_score
        tgt_cf_candidate = None
    else:
        raise ValueError('Either target rank or target score must be provided')

    return explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate


def define_explainer_HUDD(pipeline_fitness, ranker, df_qId_HUDD_pre, cols_dict_HUDD, cols_dict_FEDD, feature_dtypes, explanation_method):

    super_pipeline_model = SuperRankerPipeline(pipeline_fitness, ranker, cols_dict_FEDD)

    data_dice = dice_ml.Data(dataframe=df_qId_HUDD_pre[cols_dict_HUDD['cols_pred'] + [cols_dict_HUDD['outcome_name_col']]],
                             continuous_features=cols_dict_HUDD['continuous_features'],
                             outcome_name=cols_dict_HUDD['outcome_name_col'])

    kwargs = {'top_k': MacroVariables.TOP_K, 'features_dtype': feature_dtypes}

    model_dice = dice_ml.Model(model=super_pipeline_model,
                               backend={'explainer': 'dice_xgboost.DiceGenetic',
                                        'model': "lgbmranker_pipeline_model.LGBMRankerPipelineModel"},
                               model_type="regressor",
                               # model_type="classifier",
                               kw_args=kwargs)

    explainer = dice_ml.Dice(data_dice, model_dice, method=explanation_method)

    return explainer, data_dice, model_dice


def get_explanations_HUDD(df_qId_HUDD, exp_c, cols_dict_cf, explainer):
    c_th_lambda = df_qId_HUDD[df_qId_HUDD['pred_rank'] == MacroVariables.TOP_K].iloc[0]['lambda']
    explanations = explainer.generate_counterfactuals(exp_c['profile'][
                                                          # cols_dict_cf['cols_id'] +
                                                          cols_dict_cf['cols_pred']],
                                                      total_CFs=10,
                                                      desired_range=[c_th_lambda, 100],
                                                      # desired_class="opposite",
                                                      verbose=True)
    return explanations

def build_pipeline_fitness(df_CDS_JDS):
    maps_matching = build_matching_functions()

    # Calculation as fit-transform preprocessing
    pipeline_fitness = Pipeline(steps=[
        ("init", AttachMetadata(md_CDS_JDS_ADS)),
        ("matching", DerivedColumn(maps_matching)),
        ("end", DetachMetadata())
    ])

    fitness_matrix = pipeline_fitness.fit_transform(X=df_CDS_JDS)


    cols_dict_FEDD = define_cols_dict_FEDD()
    return pipeline_fitness, fitness_matrix, cols_dict_FEDD


def train(ranker, df_train, df_val, cols_dict):
    df_train_counts = df_train.groupby("qId")["qId"].count().to_numpy()
    df_val_counts = df_val.groupby("qId")["qId"].count().to_numpy()

    # Fitting ranker:
    ranker.fit(
        X=df_train[cols_dict['cols_pred']],
        # LightGBM relevance is the higher the better
        y=rank2relevance(df_train, MacroVariables.TOP_K, cols_dict['col_rank']),
        group = df_train_counts,
        eval_at = [MacroVariables.TOP_K],
        # LightGBM relevance is the higher the better
        eval_set =[(df_val[cols_dict['cols_pred']], rank2relevance(df_val, MacroVariables.TOP_K, cols_dict['col_rank']))],
        eval_group =[df_val_counts]
    )

    return ranker


def evaluate(ranker, df_eval, cols_dict):
    df_test_counts = df_eval.groupby("qId")["qId"].count().to_numpy()
    # Predicting ranker:
    df_eval['lambda'] = ranker.predict(df_eval[cols_dict['cols_pred']])
    df_eval['pred_rank'] = df_eval.groupby("qId")['lambda'].rank('dense', ascending=False)
    df_eval['pred_rank'] = df_eval['pred_rank'].apply(lambda x: x if x <= MacroVariables.TOP_K else MacroVariables.TOP_K + 1)

    return df_eval

def transform_split(df_train_HUDD, df_val_HUDD, df_test_HUDD, pipeline_fitness, cols_dict_HUDD):
    df_train_FEDD = pipeline_fitness.transform(df_train_HUDD)
    df_train_FEDD.reset_index(drop=True, inplace=True)
    df_train_FEDD[cols_dict_HUDD['cols_id']] = df_train_HUDD[cols_dict_HUDD['cols_id']].values
    df_train_FEDD[cols_dict_HUDD['col_rank']] = df_train_HUDD[cols_dict_HUDD['col_rank']].values

    df_val_FEDD = pipeline_fitness.transform(df_val_HUDD)
    df_val_FEDD.reset_index(inplace=True)
    df_val_FEDD[cols_dict_HUDD['cols_id']] = df_val_HUDD[cols_dict_HUDD['cols_id']].values
    df_val_FEDD[cols_dict_HUDD['col_rank']] = df_val_HUDD[cols_dict_HUDD['col_rank']].values

    df_test_FEDD = pipeline_fitness.transform(df_test_HUDD)
    df_test_FEDD.reset_index(drop=True, inplace=True)
    df_test_FEDD[cols_dict_HUDD['cols_id']] = df_test_HUDD[cols_dict_HUDD['cols_id']].values
    df_test_FEDD[cols_dict_HUDD['col_rank']] = df_test_HUDD[cols_dict_HUDD['col_rank']].values

    return df_train_FEDD, df_val_FEDD, df_test_FEDD

def ranking_pipeline(df_train_FEDD, df_val_FEDD, df_test_FEDD, cols_dict_FEDD):
    pipeline_fitness.transform(df_train_HUDD)
    # Define the ranking model
    ranker = LGBMRanker(
        objective="lambdarank",
        class_weight="balanced",
        boosting_type="gbdt",
        importance_type="gain",
        learning_rate=0.1,
        n_estimators=100,
        force_row_wise=True,
        n_jobs=-1,  # max parallelism
        verbose=-1  # no verbosity
    )

    ranker = train(ranker, df_train_FEDD, df_val_FEDD, cols_dict_FEDD)
    df_train_FEDD = evaluate(ranker, df_train_FEDD, cols_dict_FEDD)
    df_val_FEDD = evaluate(ranker, df_val_FEDD, cols_dict_FEDD)
    df_test_FEDD = evaluate(ranker, df_test_FEDD, cols_dict_FEDD)

    return ranker, df_train_FEDD, df_val_FEDD, df_test_FEDD

def attach_predictions(df_CDS_JDS, ranker, pipeline_fitness, cols_dict_FEDD):
    df_CDS_JDS['lambda'] = ranker.predict(pipeline_fitness.transform(df_CDS_JDS)[cols_dict_FEDD['cols_pred']])
    df_CDS_JDS['pred_rank'] = df_CDS_JDS.groupby("qId")['lambda'].rank('dense', ascending=False)
    return df_CDS_JDS


if __name__ == '__main__':
    args = parse_args()
    df_CDS_JDS, cols_dict_HUDD = load_dataset(fair_data=MacroVariables.FAIR_DATA)
    pipeline_fitness, df_fitness_mat, cols_dict_FEDD = build_pipeline_fitness(df_CDS_JDS)
    df_train_HUDD, df_val_HUDD, df_test_HUDD = data_split(df_CDS_JDS)
    df_train_FEDD, df_val_FEDD, df_test_FEDD = transform_split(df_train_HUDD, df_val_HUDD, df_test_HUDD, pipeline_fitness, cols_dict_HUDD)

    ranker, df_train_FEDD, df_val_FEDD, df_test_FEDD = ranking_pipeline(df_train_FEDD, df_val_FEDD, df_test_FEDD, cols_dict_FEDD)
    df_CDS_JDS = attach_predictions(df_CDS_JDS, ranker, pipeline_fitness, cols_dict_FEDD)

    df_qId_HUDD, exp_c,  = extract_explicand_data_cf(job_id=args.job_id, exp_c_pred_rank=args.candidate_position,
                                                     pipeline_fitness=pipeline_fitness, df_CDS_JDS=df_CDS_JDS)

    df_qId_HUDD_pre, feature_dtypes = prepare_data_cf(df_qId_HUDD, cols_dict_HUDD)

    explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate = define_target(args, df_qId_HUDD)

    explainer, data_dice, model_dice = define_explainer_HUDD(pipeline_fitness, ranker, df_qId_HUDD_pre,
                                                             cols_dict_HUDD, cols_dict_FEDD, feature_dtypes,
                                                             explanation_method=args.explanation_method)

    print('Counterfactual Explanations:')
    explanations_HUDD = get_explanations_HUDD(df_qId_HUDD, exp_c, cols_dict_HUDD, explainer)
    print(explanations_HUDD.visualize_as_dataframe())
