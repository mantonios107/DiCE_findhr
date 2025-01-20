import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from preprocessing import build_fitness_matrix
from MACRO import MacroVariables
from utils import rank2relevance
from lightgbm import LGBMRanker


def data_split_FEDD(df_fitness_mat):
    all_jobs = df_fitness_mat['qId'].unique()
    train_jobs, test_jobs = train_test_split(all_jobs, test_size=0.2, random_state=42, shuffle=False)
    train_jobs, val_jobs = train_test_split(train_jobs, test_size=0.25, random_state=42, shuffle=False)

    # Build train, test and validation sets, ensuring they are sorted by qId, kId
    df_train = df_fitness_mat[df_fitness_mat['qId'].isin(train_jobs)].sort_values(["qId", "kId"])
    df_val = df_fitness_mat[df_fitness_mat['qId'].isin(val_jobs)].sort_values(["qId", "kId"])
    df_test = df_fitness_mat[df_fitness_mat['qId'].isin(test_jobs)].sort_values(["qId", "kId"])

    return df_train, df_val, df_test


def init(df_fitness_mat):
    df_train, df_val, df_test = data_split_FEDD(df_fitness_mat)
    # Define subsets of columns
    cols_id = ['qId', 'kId']  # ids
    cols_pred = [  # predictive
        'fitness_Contract',
        'fitness_Nationality',
        'fitness_Education',
        'fitness_Experience',
        # 'fitness_Age',
        'fitness_Gender',
        'fitness_Languages',
        'fitness_Competences',
        'fitness_Knowledge']
    cols_sensitive = ['Gender_c']  # sensitive attribute(s)
    col_target = 'score'  # target value for ranking
    col_rank = 'rank'  # rank value for ranking

    cols_dict_FEDD = {'cols_id': cols_id,
                      'cols_pred': cols_pred,
                      'cols_sensitive': cols_sensitive,
                      'col_target': col_target,
                      'col_rank': col_rank}
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
    return ranker, df_train, df_val, df_test, cols_dict_FEDD


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


def evaluate(ranker, df_test, cols_dict):
    df_test_counts = df_test.groupby("qId")["qId"].count().to_numpy()
    # Predicting ranker:
    df_test['lambda'] = ranker.predict(df_test[cols_dict['cols_pred']])
    df_test['pred_rank'] = df_test.groupby("qId")['lambda'].rank('dense', ascending=False)
    df_test['pred_rank'] = df_test['pred_rank'].apply(lambda x: x if x <= MacroVariables.TOP_K else MacroVariables.TOP_K + 1)

    return df_test


def ranking_pipeline(df_fitness_mat):
    ranker, df_train, df_val, df_test, cols_dict_FEDD = init(df_fitness_mat)
    ranker = train(ranker, df_train, df_val, cols_dict_FEDD)
    df_test = evaluate(ranker, df_test, cols_dict_FEDD)
    return ranker, df_test, cols_dict_FEDD