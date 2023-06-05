from zipfile import ZipFile

import os
import pandas as pd


current_path = os.path.dirname(os.path.abspath(__file__))

class TitanicDatasetLoader:

    def __init__(self):
        # Specify the path to the zip file        
        self.zip_path = os.path.join(current_path, '../../', 'titanic.zip')

    def __read(self, dataset_file_name):
        # Open the zip file
        with ZipFile(self.zip_path, 'r') as zip_file:
            # Extract the CSV file from the zip
            csv_file = zip_file.extract(dataset_file_name)

        # Read the CSV file using pandas
        return pd.read_csv(csv_file)

    def read_train_dataset(self):
        return self.__read(dataset_file_name='train.csv')
        
    
    def read_test_dataset(self):
        return self.__read(dataset_file_name='test.csv')
