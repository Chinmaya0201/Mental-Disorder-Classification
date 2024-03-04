import os
import sys 

import pandas as pd 
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():

    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')


class DataIngestion():

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info('Started Data Ingestion')

        try:

            # Reading the dataset
            df = pd.read_csv(r'Data\Dataset-Mental-Disorders.csv')
            logging.info('read the dataset')

            # Making artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            logging.info("created artifacts folder")

            #Saving dataframe in raw data
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)
            logging.info('Saving dataframe in raw data')

            #Train test split
            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            logging.info("Train test split Done")

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)
            logging.info('Saving Train csv')

            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)
            logging.info('Saving Test csv')

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e)

if __name__ == "__main__":

    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()
    print(train_path, test_path)

    preprocessing_obj = DataPreprocessing()
    new_train_path , new_test_path = preprocessing_obj.initiate_data_preprocessing(train_path, test_path)
    print(new_train_path, new_test_path)

    data_transformation = DataTransformation()
    train_arr, test_arr, obj_path = data_transformation.initiate_data_transformation(new_train_path, new_test_path)
    print('data transformation')
    print("\ntrain array=",train_arr)
    print("\ntest array=",test_arr)
    print("\nobj path=", obj_path)

    model_trainer = ModelTrainer()
    print('MOdel Trainer')
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))



