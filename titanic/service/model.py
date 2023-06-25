import lightgbm as lgb
import random

from abc import ABC, abstractmethod
from pandas import DataFrame



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

    def __init__(self, train_percentage: int = None):

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
        self.invalid_columns = ['PassenderId', 'Name', 'Ticket', 'Cabin', 'Embarked']

        self.valid_dataset = None

        self.X_train = None
        self.y_train = None

        self.X_valid = None
        self.y_valid = None

        self.train_percentage: int | None = train_percentage

        self.dataset = None
        self.columns: list = []

    def __pre_processing__(self):

        for invalid_column in self.invalid_columns:
            if invalid_column in self.dataset.columns:
                del self.dataset[invalid_column]

        self.dataset['Sex'].replace('male', 0, inplace=True)
        self.dataset['Sex'].replace('female', 1, inplace=True)

    def set_current_dataset(self, dataset: DataFrame) -> None:
        assert isinstance(dataset, DataFrame)
        self.dataset = dataset
        self.columns = self.dataset.columns

    def split_into_train_valid(self):

        if self.train_percentage is not None:

            assert 0 < self.train_percentage < 1

            upper_bound = int(self.dataset.shape[0] * self.train_percentage)

            self.train_dataset = self.dataset.iloc[:upper_bound]
            self.valid_dataset = self.dataset.iloc[upper_bound:]
        
        else:

            self.train_dataset = self.dataset
            self.valid_dataset = self.dataset

    def split_into_input_output(self):

        if self.response_column in self.train_dataset.columns:
            self.y_train = self.train_dataset[self.response_column]
            del self.train_dataset[self.response_column]

        if self.response_column in self.valid_dataset.columns:
            self.y_valid = self.valid_dataset[self.response_column]
            del self.valid_dataset[self.response_column]

        self.X_train = self.train_dataset
        self.X_valid = self.valid_dataset


    def train(self, dataset):
        
        self.set_current_dataset(dataset)
        self.__pre_processing__()

        self.split_into_train_valid()
        self.split_into_input_output()

        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_valid = lgb.Dataset(self.X_valid, self.y_valid)

        self.model = lgb.train(params=self.params, train_set=lgb_train, valid_sets=[lgb_valid], num_boost_round=500, early_stopping_rounds=5, verbose_eval=3)

    def predict(self, dataset):

        result = dataset[['PassengerId']]

        self.set_current_dataset(dataset)
        self.__pre_processing__()

        y_pred = self.model.predict(dataset, num_iteration=self.model.best_iteration)
        result['Survived'] = [1 if y >= .5 else 0 for y in y_pred]

        return result
    