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

import lightgbm as lgb

class LightGBMModel(Model):

    def __init__(self):
        self.model = None
        self.params: dict = {}

    def train(self, dataset):
        self.model = lgb.train(params=self.params, train_set=dataset)

    def predict(self, dataset):
        return self.model 