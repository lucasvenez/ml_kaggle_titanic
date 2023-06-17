from titanic.service import TitanicDatasetLoader
from titanic.service import Model, RandomModel, LightGBMModel

import numpy as np


def submit_to_kaggle(result_file_path, message=''):

    import subprocess

    COMPETITION_NAME = 'titanic'

    # Submit the file using Kaggle API command
    command = f'kaggle competitions submit -c {COMPETITION_NAME} -f {result_file_path} -m "{message}"'
    
    return subprocess.run(command, shell=True)


if __name__ == '__main__':
    
    loader: TitanicDatasetLoader = TitanicDatasetLoader()
    train_dataset = loader.read_train_dataset()

    y = train_dataset['Survived']

    m1: Model = LightGBMModel()
    m1.train(train_dataset)

    test_dataset = loader.read_test_dataset()
    result = m1.predict(test_dataset)

    print(result)

    if True:

        result_file_path = 'output.csv'
        result.to_csv(result_file_path, sep=',', header=True, index=False)
        submit_to_kaggle(result_file_path, message=f'{m1.__class__.__name__} v{m1.__version__}')
    
    else:
        y_pred = np.array(m1.predict(m1.X_validation)['Survived'].values)
        accuracy = sum(m1.y_validation.values == y_pred) / len(y_pred)

        print('\n==========================================')
        print(f'Accuracy: {round(accuracy * 100, 2)}%')
        print('==========================================\n')
