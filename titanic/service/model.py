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
            'verbose': 1,
            'max_bin': 10
        }

        self.response_column = 'Survived'
        self.invalid_columns = ['PassenderId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', self.response_column]

        self.validation_dataset = None
        self.X_validation = None
        self.y_validation = None

    def train(self, dataset, train_percentage=None):
        
        if train_percentage is not None:

            assert 0 < train_percentage < 1

            upper_bound = int(dataset.shape[0] * train_percentage)

            train_dataset = dataset.iloc[:upper_bound]
            self.validation_dataset = dataset.iloc[upper_bound:]
        
        else:

            train_dataset = dataset
            self.validation_dataset = dataset

        y = train_dataset[self.response_column]
        self.y_validation = self.validation_dataset[self.response_column]

        for invalid_column in self.invalid_columns:
            if invalid_column in dataset.columns:
                try:
                    del train_dataset[invalid_column]
                    del self.validation_dataset[invalid_column]
                except:
                    pass

        X = train_dataset
        self.X_validation = self.validation_dataset

        lgb_train = lgb.Dataset(X, y)
        lgb_valid = lgb.Dataset(self.X_validation, self.y_validation)
        self.model = lgb.train(params=self.params, train_set=lgb_train, valid_sets=[lgb_valid], num_boost_round=500, early_stopping_rounds=5, verbose_eval=3)

    def predict(self, dataset):

        result = dataset[['PassengerId']]

        for invalid_column in self.invalid_columns:
            if invalid_column in dataset.columns:
                del dataset[invalid_column]

        y_pred = self.model.predict(dataset, num_iteration=self.model.best_iteration)
        result['Survived'] = [1 if y >= .5 else 0 for y in y_pred]

        return result
    