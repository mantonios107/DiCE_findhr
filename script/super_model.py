from sklearn.pipeline import Pipeline
from cf_exp_FEDD import (load_dataset, build_fitness_matrix, ranking_pipeline)
from MACRO import MacroVariables

class SuperRankerPipeline:

    def __init__(self, pipeline, ranker, cols_dict_FEDD): # steps, *, memory=None, verbose=False):
        self.pipeline = pipeline
        self.ranker = ranker
        self.cols_dict_FEDD = cols_dict_FEDD

    def predict(self, X):
        # params_pipeline = {k: v for k, v in params.items() if k in self.pipeline.get_params().keys()}
        _intermediate = self.pipeline.transform(X)
        # print('_intermediate', _intermediate)
        # _intermediate.drop(columns=['qId', 'kId'], inplace=True)
        return self.ranker.predict(_intermediate[self.cols_dict_FEDD['cols_pred']])

    # def fit(self, X, y, **params):
    #     _intermediate = self.pipeline.fit_transform(X)
    #      = self.pipeline.transform(X)
    #     self.ranker.fit(_intermediate[self.cols_dict_FEDD['cols_pred']], y)

if __name__ == '__main__':
    df_CDS_JDS, cols_dict_HUDD = load_dataset(fair_data=MacroVariables.FAIR_DATA)
    pipeline_fitness, df_fitness_mat = build_fitness_matrix(df_CDS_JDS,
                                                            cols_dict_HUDD,
                                                            fair_data=MacroVariables.FAIR_DATA)
    ranker, df_test, cols_dict_FEDD = ranking_pipeline(df_fitness_mat)
    super_pipeline = SuperRankerPipeline(pipeline_fitness, ranker, cols_dict_FEDD)
    print('super_pipeline', super_pipeline)
    print('Predict')
    df_CDS_JDS['lambda'] = super_pipeline.predict(df_CDS_JDS)

