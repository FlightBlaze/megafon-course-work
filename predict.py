import numpy as np
import pandas as pd
import dask.dataframe as dd
import pickle
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


class Ensemble(BaseEstimator, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y = None):
        return self

    def predict_proba(self, X):
        n_models = len(self.models)
        preds = []
        for i in range(n_models):
            preds.append(self.models[i].predict_proba(X))
        
        p = np.zeros((X.shape[0], 2))
        for i in range(p.shape[0]):
            for j in range(2):
                best_proba = 0
                mean_proba = 0
                for k in range(n_models):
                    proba = preds[k][i,j]
                    if proba > best_proba:
                        best_proba = proba
                    mean_proba += proba
                mean_proba /= k
                p[i,j] = best_proba
        
        return p

    def predict(self, X):
        return self.predict_proba(X)[:,1] > 0.037241


models = []
num_models = 4
for i in range(num_models):
    models.append(pickle.load(open(f'my_data/models/{i}.pickle', 'rb')))

ens = Ensemble(models)


class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X = None, y = None):
        features = dd.read_csv('data/features.csv', sep='\t').drop(['Unnamed: 0'], axis=1)
        data_test = dd.read_csv('data/data_test.csv').drop(['Unnamed: 0'], axis=1)

        return [data_test, features]


loader = DataLoader()


class MergeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        data = X[0]
        if 'target' in data.columns:
            data = data.drop('target', axis=1)
            # raise ValueError('Please remove the target feature from the data (first DataFrame)')
        if 'Unnamed: 0' in data.columns:
            raise ValueError('Please remove the "Unnamed: 0" feature from the data (first DataFrame)')
        
        feats = X[1]
        if 'Unnamed: 0' in feats.columns:
            raise ValueError('Please remove the "Unnamed: 0" feature from the features (second DataFrame)')
        
        ids = np.unique(data['id'])
        feats = feats[feats['id'].isin(ids)].compute()
        data = data.compute()

        feats.sort_values(by='id', inplace=True)
        data.sort_values(by='id', inplace=True)

        merged = pd.merge_asof(data, feats, by='buy_time', on='id', direction='nearest')

        return merged


class BeforePredictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X.drop('id', axis=1)


merger = MergeTransformer()


class BeforePredictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X.drop('id', axis=1)


before_pred = BeforePredictTransformer()


class PredSaver(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        target = X[0][:, 1]
        df = X[1]
        answers = pd.DataFrame()
        answers['id'] = df['id']
        answers['vas_id'] = df['vas_id']
        answers['buy_time'] = df['buy_time']
        answers['target'] = target
        answers.to_csv('my_data/answers_test.csv', index=False)
        return X


class Printer(BaseEstimator, TransformerMixin):
    def __init__(self, message):
        self.message = message

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        print(self.message)
        return X

class ProbPredictor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def predict_proba(self, X, y = None):
        self.load_pipe = Pipeline([
            ('print_1', Printer('Loading data...')),
            ('loader', loader),
            ('print_2', Printer('Merging features with data...')),
            ('merger', merger)
        ])

        self.pred_pipe = Pipeline([
            ('before_pred', before_pred),
            ('print', Printer('Predicting...')),
            ('classifier', ens)
        ])

        self.saver = Pipeline([
            ('print_1', Printer('Saving results...')),
            ('saver', PredSaver()),
            ('print_2', Printer('Done!'))
        ])

        data = self.load_pipe.transform(X)
        pred_proba = self.pred_pipe.predict_proba(data)
        self.saver.transform((pred_proba, data))

        return None


pipe = Pipeline([
    ('predictor', ProbPredictor())
])

pipe.predict_proba(None)
