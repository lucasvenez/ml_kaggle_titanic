from titanic.service import TitanicDatasetLoader
from titanic.service import Model, RandomModel, LightGBMModel


def submit_to_kaggle(result_file_path, message=''):

    import subprocess

    COMPETITION_NAME = 'titanic'

    # Submit the file using Kaggle API command
    command = f'kaggle competitions submit -c {COMPETITION_NAME} -f {result_file_path} -m "{message}"'
    
    return subprocess.run(command, shell=True)


if __name__ == '__main__':
    
    loader: TitanicDatasetLoader = TitanicDatasetLoader()
    train_dataset = loader.read_train_dataset()

    m1: Model = LightGBMModel()
    m1.train(train_dataset)

    test_dataset = loader.read_test_dataset()
    result = m1.predict(test_dataset)

    print(result)

    if True:
        result_file_path = 'output.csv'
        result.to_csv(result_file_path, sep=',', header=True, index=False)
        submit_to_kaggle(result_file_path, message=f'{m1.__class__.__name__} v{m1.__version__}')
