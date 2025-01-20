import argparse
import dice_ml
from MACRO import MacroVariables
from ranking import ranking_pipeline
from preprocessing import load_dataset, build_fitness_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', '-j', type=int, default=161, # 160-199
                        help='The job id for which the counterfactual explanation is to be generated')

    parser.add_argument('--candidate_position', '-c', type=int, default=15, # 16
                        help='The position of the candidate in the ranking for which the counterfactual explanation is to be generated')

    parser.add_argument('--explanation_method', '-m', type=str, choices=['random', 'genetic', 'kdtree'], default='genetic')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--target_rank', '-r', type=int, default=MacroVariables.TOP_K,
                        help='The target rank for the counterfactual explanation')
    group.add_argument('--target_score', '-s', type=float)

    return parser.parse_args()


def define_cols_dict():
    outcome_name_col = 'lambda'  # 'pred_rank'
    continuous_features = ['fitness_Languages', 'fitness_Competences',
                           'fitness_Knowledge']  # ['Age_c', 'Experience_c'],
    categorical_features = ['fitness_Contract', 'fitness_Nationality', 'fitness_Education', 'fitness_Experience',
                            # 'fitness_Age',
                            'fitness_Gender']
    cols_pred = continuous_features + categorical_features
    return {'outcome_name_col': outcome_name_col, 'continuous_features': continuous_features,
            'categorical_features': categorical_features, 'cols_pred': cols_pred}


def extract_explicand_data_cf(job_id, candidate_position, df_test, ranker, cols_dict_FEDD, df_CDS_JDS):
    # df_qId contains the data for the job qId
    df_qId_FEDD = df_test[df_test['qId'] == job_id]
    df_qId_FEDD['lambda'] = ranker.predict(df_qId_FEDD[cols_dict_FEDD['cols_pred']])
    df_qId_FEDD['pred_rank'] = df_qId_FEDD.groupby("qId")['lambda'].rank('dense', ascending=False)

    exp_c_pred_rank = candidate_position

    # Extract the explicand candidate kId
    exp_c_kId = df_qId_FEDD.loc[df_qId_FEDD['pred_rank'] == exp_c_pred_rank, 'kId'].iloc[0]

    # Isolate the candidates' profiles applying for the job qId
    df_qId_HUDD = df_CDS_JDS[df_CDS_JDS['qId'] == job_id]

    # Isolate the explicand candidate profile
    exp_c_profile = df_CDS_JDS[df_CDS_JDS['kId'] == exp_c_kId]

    exp_c = {'kId': exp_c_kId, 'profile': exp_c_profile}

    cols_dict = define_cols_dict()

    return df_qId_FEDD, df_qId_HUDD, exp_c, cols_dict


def prepare_data_cf(df_qId_FEDD, cols_dict):

    # Convert data types
    df_qId_FEDD_pre = df_qId_FEDD[cols_dict['categorical_features']].astype('int').copy(deep=True)
    df_qId_FEDD_pre[cols_dict['continuous_features']] = df_qId_FEDD[cols_dict['continuous_features']].astype('float').copy(deep=True)
    df_qId_FEDD_pre[cols_dict['outcome_name_col']] = df_qId_FEDD[cols_dict['outcome_name_col']].copy(deep=True)
    feature_dtypes = {col: df_qId_FEDD_pre[col].dtype for col in df_qId_FEDD_pre[cols_dict['cols_pred']].columns}

    return df_qId_FEDD_pre, feature_dtypes


def define_target(args, df_qId_FEDD):
    # 'in_top_k' or 'out_top_k' depending on the candidate position
    explicand_class = 'in_top_k' if args.candidate_position <= MacroVariables.TOP_K else 'out_top_k'

    # target rank for counterfactual explanation
    if args.target_rank:
        tgt_cf_rank = args.target_rank
        tgt_cf_score = df_qId_FEDD[df_qId_FEDD['pred_rank'] == tgt_cf_rank]['score'].iloc[0]
        tgt_cf_candidate = df_qId_FEDD[df_qId_FEDD['pred_rank'] == tgt_cf_rank]

    elif args.target_score:
        tgt_cf_rank = None
        tgt_cf_score = args.target_score
        tgt_cf_candidate = None
    else:
        raise ValueError('Either target rank or target score must be provided')

    return explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate


def define_explainer_FEDD(ranker, df_qId_FEDD_pre, cols_dict_cf, feature_dtypes, explanation_method):
    data_dice = dice_ml.Data(dataframe=df_qId_FEDD_pre[cols_dict_cf['cols_pred'] + [cols_dict_cf['outcome_name_col']]],
                             continuous_features=cols_dict_cf['continuous_features'],
                             categorical_features=cols_dict_cf['categorical_features'],
                             outcome_name=cols_dict_cf['outcome_name_col'])

    kwargs = {'top_k': MacroVariables.TOP_K, 'features_dtype': feature_dtypes}

    model_dice = dice_ml.Model(model=ranker,
                               backend={'explainer': 'dice_xgboost.DiceGenetic',
                                        'model': "lgbmranker_model.LGBMRankerModel"},
                               model_type="regressor",
                               # model_type="classifier",
                               kw_args=kwargs)

    explainer = dice_ml.Dice(data_dice, model_dice, method=explanation_method)

    return explainer, data_dice, model_dice


def get_explanations_FEDD(df_qId_FEDD, exp_c, cols_dict_cf, explainer):

    c_th_lambda = df_qId_FEDD[df_qId_FEDD['pred_rank'] == MacroVariables.TOP_K].iloc[0]['lambda']
    explanations = explainer.generate_counterfactuals(exp_c['profile'][cols_dict_cf['cols_pred']],
                                                      total_CFs=10,
                                                      desired_range=[c_th_lambda, 100],
                                                      # desired_class="opposite",
                                                      verbose=True)
    return explanations


if __name__ == '__main__':

    args = parse_args()
    df_CDS_JDS, cols_dict_HUDD = load_dataset(fair_data=MacroVariables.FAIR_DATA)
    pipeline_fitness, df_fitness_mat = build_fitness_matrix(df_CDS_JDS, cols_dict_HUDD, fair_data=MacroVariables.FAIR_DATA)
    ranker, df_test, cols_dict_FEDD = ranking_pipeline(df_fitness_mat)

    df_qId_FEDD, df_qId_HUDD, exp_c, cols_dict_cf = extract_explicand_data_cf(args.job_id, args.candidate_position, df_test, ranker, cols_dict_FEDD, df_CDS_JDS)
    # Convert data types
    df_qId_FEDD_pre, feature_dtypes = prepare_data_cf(df_qId_FEDD, cols_dict_cf)

    explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate = define_target(args, df_qId_FEDD)

    explainer, data_dice, model_dice = define_explainer_FEDD(ranker, df_qId_FEDD_pre, cols_dict_cf, feature_dtypes, args.explanation_method)
    print('Explanations for the counterfactuals:')
    explanations_FEDD = get_explanations_FEDD(df_qId_FEDD, exp_c, cols_dict_cf, explainer)
    print(explanations_FEDD.visualize_as_dataframe())

    explanations_FEDD.cf_examples_list[0].final_cfs_df.to_csv('final_cfs_df_FEDD.csv')
