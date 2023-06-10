import lightgbm as lgb
import random

from abc import ABC, abstractmethod



class Model(ABC):
    
    @abstractmethod
    def train(self, dataset):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset):
        raise NotImplementedError()


class RandomModel(Model):

    __version__ = '0.0.1'

    def train(self, dataset):
        self.r = random

    def predict(self, dataset):

        dataset = dataset.copy()

        for column in dataset.columns:
            if column != 'PassengerId':
                del dataset[column]
        
        survived = []

        for _ in range(dataset.shape[0]):
            survived.append(self.r.randint(0, 1))

        dataset['Survived'] = survived

        return dataset


class LightGBMModel(Model):

    __version__ = '0.0.1'

    def __init__(self):

        self.model = None

        self.params: dict = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 90,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1
        }

        self.response_column = 'Survived'
        self.invalid_columns = ['PassenderId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', self.response_column]

    def train(self, dataset):
        
        y = dataset[self.response_column]

        for invalid_column in self.invalid_columns:
            if invalid_column in dataset.columns:
                del dataset[invalid_column]

        X = dataset

        lgb_train = lgb.Dataset(X, y)
        self.model = lgb.train(params=self.params, train_set=lgb_train, num_boost_round=500)

    def predict(self, dataset):

        result = dataset[['PassengerId']]

        for invalid_column in self.invalid_columns:
            if invalid_column in dataset.columns:
                del dataset[invalid_column]

        y_pred = self.model.predict(dataset, num_iteration=self.model.best_iteration)
        result['Survived'] = [1 if y >= .5 else 0 for y in y_pred]

        return result
    