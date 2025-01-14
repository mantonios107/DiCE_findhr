import streamlit as st
import pandas as pd
import numpy as np
import dice_ml
from MACRO import MacroVariables
from ranking import ranking_pipeline
from preprocessing import load_dataset, build_fitness_matrix

# Define necessary functions (assuming they are available in your environment)
def define_cols_dict():
    outcome_name_col = 'lambda'  # 'pred_rank'
    continuous_features = ['fitness_Languages', 'fitness_Competences',
                           'fitness_Knowledge']
    categorical_features = ['fitness_Contract', 'fitness_Nationality', 'fitness_Education', 'fitness_Experience',
                            'fitness_Age', 'fitness_Gender']
    cols_pred = continuous_features + categorical_features
    cols_id = ['qId', 'kId']
    return {'outcome_name_col': outcome_name_col, 'continuous_features': continuous_features,
            'categorical_features': categorical_features, 'cols_pred': cols_pred, 'cols_id': cols_id}

def extract_explicand_data_cf(job_id, candidate_position, df_test, ranker, cols_dict_FEDD, df_CDS_JDS):
    # df_qId contains the data for the job qId
    df_qId_FEDD = df_test[df_test['qId'] == job_id].copy()
    df_qId_FEDD['lambda'] = ranker.predict(df_qId_FEDD[cols_dict_FEDD['cols_pred']])
    df_qId_FEDD['pred_rank'] = df_qId_FEDD.groupby("qId")['lambda'].rank('dense', ascending=False)

    exp_c_pred_rank = candidate_position

    # Extract the explicand candidate kId
    exp_c_kId = df_qId_FEDD.loc[df_qId_FEDD['pred_rank'] == exp_c_pred_rank, 'kId'].iloc[0]

    # Isolate the candidates' profiles applying for the job qId
    df_qId_HUDD = df_CDS_JDS[df_CDS_JDS['qId'] == job_id]

    # Isolate the explicand candidate profile
    exp_c_profile = df_CDS_JDS[df_CDS_JDS['kId'] == exp_c_kId]

    exp_c = {'kId': exp_c_kId, 'profile': exp_c_profile, 'pred_rank': exp_c_pred_rank, 'pred_lambda': df_qId_FEDD.loc[df_qId_FEDD['kId'] == exp_c_kId, 'lambda'].iloc[0]}

    cols_dict = define_cols_dict()

    return df_qId_FEDD, df_qId_HUDD, exp_c, cols_dict

def prepare_data_cf(df_qId_FEDD, cols_dict):
    # Convert data types
    df_qId_FEDD_pre = df_qId_FEDD[cols_dict['categorical_features']].astype('int').copy(deep=True)
    df_qId_FEDD_pre[cols_dict['continuous_features']] = df_qId_FEDD[cols_dict['continuous_features']].astype('float').copy(deep=True)
    df_qId_FEDD_pre[cols_dict['outcome_name_col']] = df_qId_FEDD[cols_dict['outcome_name_col']].copy(deep=True)
    feature_dtypes = {col: df_qId_FEDD_pre[col].dtype for col in df_qId_FEDD_pre[cols_dict['cols_pred']].columns}

    return df_qId_FEDD_pre, feature_dtypes

def define_target(candidate_position, target_rank, df_qId_FEDD):
    # 'in_top_k' or 'out_top_k' depending on the candidate position
    explicand_class = 'in_top_k' if candidate_position <= MacroVariables.TOP_K else 'out_top_k'

    # target rank for counterfactual explanation
    tgt_cf_rank = target_rank
    tgt_cf_score = df_qId_FEDD[df_qId_FEDD['pred_rank'] == tgt_cf_rank]['lambda'].iloc[0]
    tgt_cf_candidate = df_qId_FEDD[df_qId_FEDD['pred_rank'] == tgt_cf_rank]

    return explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate

def define_explainer(ranker, df_qId_FEDD_pre, cols_dict_cf, feature_dtypes, explanation_method):
    data_dice = dice_ml.Data(dataframe=df_qId_FEDD_pre[cols_dict_cf['cols_pred'] + [cols_dict_cf['outcome_name_col']]],
                     continuous_features=cols_dict_cf['continuous_features'],
                     outcome_name=cols_dict_cf['outcome_name_col'])

    kwargs = {'top_k': MacroVariables.TOP_K, 'features_dtype': feature_dtypes}

    model_dice = dice_ml.Model(model=ranker,
                               backend={'explainer': 'dice_xgboost.DiceGenetic',
                                        'model': "lgbmranker_model.LGBMRankerModel"},
                               model_type="regressor",
                               kw_args=kwargs)

    explainer = dice_ml.Dice(data_dice, model_dice, method=explanation_method)

    return explainer, data_dice, model_dice

def get_explanations(df_qId_FEDD, exp_c, cols_dict_cf, explainer):
    c_th_lambda = df_qId_FEDD[df_qId_FEDD['pred_rank'] == MacroVariables.TOP_K].iloc[0]['lambda']
    explanations = explainer.generate_counterfactuals(exp_c['profile'][cols_dict_cf['cols_pred']],
                                                      total_CFs=10,
                                                      desired_range=[c_th_lambda, 100],
                                                      verbose=True)
    return explanations

@st.cache_data
def load_datasets():
    df_CDS_JDS, cols_dict_HUDD = load_dataset(fair_data=MacroVariables.FAIR_DATA)
    pipeline_fitness, df_fitness_mat = build_fitness_matrix(df_CDS_JDS, cols_dict_HUDD, fair_data=MacroVariables.FAIR_DATA)
    ranker, df_test, cols_dict_FEDD = ranking_pipeline(df_fitness_mat)
    return df_CDS_JDS, cols_dict_HUDD, pipeline_fitness, df_fitness_mat, ranker, df_test, cols_dict_FEDD

# Streamlit App
st.title('Counterfactual Explanations for Job Candidate Rankings')

# Sidebar for user inputs
with st.sidebar.form('input_form'):
    job_id = st.number_input('Job ID', min_value=160, max_value=199, value=161, step=1)
    candidate_position = st.number_input('Candidate Position', min_value=1, value=15, step=1)
    target_rank = st.number_input('Target Rank', min_value=1, value=10, step=1)
    explanation_method = st.selectbox('Explanation Method', options=['random', 'genetic', 'kdtree'], index=1)
    submitted = st.form_submit_button('Compute Explanation')

if submitted:
    # Load datasets
    df_CDS_JDS, cols_dict_HUDD, pipeline_fitness, df_fitness_mat, ranker, df_test, cols_dict_FEDD = load_datasets()

    # Extract data for candidate
    df_qId_FEDD, df_qId_HUDD, exp_c, cols_dict_cf = extract_explicand_data_cf(job_id, candidate_position, df_test, ranker, cols_dict_FEDD, df_CDS_JDS)
    df_qId_FEDD_pre, feature_dtypes = prepare_data_cf(df_qId_FEDD, cols_dict_cf)

    # Define target
    explicand_class, tgt_cf_rank, tgt_cf_score, tgt_cf_candidate = define_target(candidate_position, target_rank, df_qId_FEDD)

    # Define explainer
    explainer, data_dice, model_dice = define_explainer(ranker, df_qId_FEDD_pre, cols_dict_cf, feature_dtypes, explanation_method)

    # Get explanations
    explanations = get_explanations(df_qId_FEDD, exp_c, cols_dict_cf, explainer)

    # Get the explanations as a dataframe
    explanations_df = explanations.cf_examples_list[0].final_cfs_df

    st.write('Counterfactual explanation for the candidate at position', candidate_position, 'for job ID', job_id)
    st.write('Target rank:', tgt_cf_rank, 'with score:', tgt_cf_score)
    st.write('Candidate profile lambda: ', exp_c['pred_lambda'])
    # Display the original candidate profile
    st.subheader('Original Candidate Profile')
    original_profile = exp_c['profile'][cols_dict_cf['cols_id'] + cols_dict_cf['cols_pred']]
    st.dataframe(original_profile)

    # Display the counterfactual explanations
    st.subheader('Counterfactual Explanations')

    # Highlight differences between original and counterfactuals
    def highlight_differences(explanations_df, original_profile, outcome_name_col='lambda'):
        df_combined = explanations_df.copy()
        print('Original Profile')
        print(original_profile.columns)
        print(original_profile.iloc[0])
        print('Explanations')
        print(explanations_df.columns)
        print(explanations_df.iloc[0])
        print('----------')
        list_cols = list(explanations_df.columns)
        list_cols.remove(outcome_name_col)
        for column in list_cols:
            df_combined[column] = np.where(explanations_df[column] != original_profile.iloc[0][column],
                                           explanations_df[column].astype(str) + ' ⬅',
                                           explanations_df[column])
        return df_combined

    explanations_highlighted = highlight_differences(explanations_df, original_profile)
    st.dataframe(explanations_highlighted)

else:
    st.write('Please enter the input parameters and press "Compute Explanation".')
