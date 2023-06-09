from titanic.service import TitanicDatasetLoader
from titanic.service import Model, RandomModel, LightGBMModel, TitanicPreProcessing

import argparse
import numpy as np


def submit_to_kaggle(result_file_path, message=''):

    import subprocess

    COMPETITION_NAME = 'titanic'

    # Submit the file using Kaggle API command
    command = f'kaggle competitions submit -c {COMPETITION_NAME} -f {result_file_path} -m "{message}"'
    
    return subprocess.run(command, shell=True)


if __name__ == '__main__':
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Training and submitting')

    # Add command-line arguments
    parser.add_argument('--kaggle', action='store_true', help='Send prediction to kaggle')
    parser.add_argument('--train-percentage', help='Train percentage', type=float)

    # Parse the arguments
    args = parser.parse_args()

    # Access the values of the arguments
    should_submit = args.kaggle
    train_percentage = args.train_percentage
    
    if isinstance(train_percentage, float) and (train_percentage <= 0 or train_percentage >= 1):
        train_percentage = None

    pre_processing: TitanicPreProcessing = TitanicPreProcessing()
    loader: TitanicDatasetLoader = TitanicDatasetLoader()
    train_dataset = loader.read_train_dataset()
    pre_processing.fit(train_dataset)

    y = train_dataset['Survived']

    m1: Model = LightGBMModel(train_percentage=train_percentage)
    m1.train(train_dataset)

    test_dataset = loader.read_test_dataset()
    pre_processing.fit(test_dataset)
    result = m1.predict(test_dataset)

    print(result)

    if should_submit:

        result_file_path = 'output.csv'
        result.to_csv(result_file_path, sep=',', header=True, index=False)
        submit_to_kaggle(result_file_path, message=f'{m1.__class__.__name__} v{m1.__version__}')
    
    else:
        y_pred = np.array(m1.predict(m1.X_valid)['Survived'].values)
        accuracy = sum(m1.y_valid.values == y_pred) / len(y_pred)

        print('\n==========================================')
        print(f'Accuracy: {round(accuracy * 100, 2)}%')
        print('==========================================\n')
