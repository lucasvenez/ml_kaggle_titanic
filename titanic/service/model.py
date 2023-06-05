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

    def train(self, dataset):
        self.r = random

    def predict(self, dataset):

        dataset = dataset.copy()

        for column in dataset.columns:
            if column != 'PassengerId':
                del dataset[column]
        
        survived = []

        for _ in range(dataset.shape[1]):
            survived.append(self.r.randint(0, 1))

        dataset['Survived'] = survived

        return dataset
    