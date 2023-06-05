from titanic.service import TitanicDatasetLoader
from titanic.service import Model, RandomModel


if __name__ == '__main__':
    
    loader: TitanicDatasetLoader = TitanicDatasetLoader()
    train_dataset = loader.read_train_dataset()

    m1: Model = RandomModel()
    m1.train(train_dataset)

    test_dataset = loader.read_test_dataset()
    result = m1.predict(test_dataset)

    print(result)
