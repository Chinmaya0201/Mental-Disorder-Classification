import os
import sys 

import pandas as pd 
import numpy as np 

from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder

from src.logger import logging 
from src.exception import CustomException 
from src.utils import convert_to_number_string, convert_to_number_yes_no

@dataclass
class DataPreprocessingConfig():

    train_data_processed_path : str = os.path.join('artifacts','train_processed.csv')
    test_data_processed_path : str = os.path.join('artifacts','test_processed.csv')

class DataPreprocessing():

    def __init__(self):

        self.data_preprocessing_config = DataPreprocessingConfig()

    def initiate_data_preprocessing(self, train_path, test_path):

        try:

            logging.info('Data Preprocessing Started')

            ## ------------Independent Variable preprocessing -----------------------------------------------

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')

            train_df['Sexual Activity'] = train_df['Sexual Activity'].apply(convert_to_number_string)
            train_df['Concentration'] = train_df['Concentration'].apply(convert_to_number_string)
            train_df['Optimisim'] = train_df['Optimisim'].apply(convert_to_number_string)

            test_df['Sexual Activity'] = test_df['Sexual Activity'].apply(convert_to_number_string)
            test_df['Concentration'] = test_df['Concentration'].apply(convert_to_number_string)
            test_df['Optimisim'] = test_df['Optimisim'].apply(convert_to_number_string)

            logging.info('Convert three column in for number of out 10 into numerical column')

            columns_to_change = ['Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation',
                                 'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down',
                                 'Admit Mistakes', 'Overthinking','Mood Swing']
            
            for col in columns_to_change:
                train_df[col] = train_df[col].apply(convert_to_number_yes_no)
                test_df[col] = test_df[col].apply(convert_to_number_yes_no)

            logging.info('Convert yes/no column into numerical column')

            ## ---------------Dependent Variable preprocessing -----------------------------------------------
                
            target_variable = 'Expert Diagnose'

            label_encoder = LabelEncoder()

            train_df[target_variable] = label_encoder.fit_transform(train_df[target_variable])
            test_df[target_variable] = label_encoder.fit_transform(test_df[target_variable])

            logging.info('Convert dependent variable into into numerical column')

            train_df.to_csv(self.data_preprocessing_config.train_data_processed_path, index= False, header= True)
            test_df.to_csv(self.data_preprocessing_config.test_data_processed_path, index= False, header= True)

            logging.info('Saved train and test dataframe into csv files')

            return (
                self.data_preprocessing_config.train_data_processed_path,
                self.data_preprocessing_config.test_data_processed_path
            )

        except Exception as e:

            logging.info(e)
            raise CustomException(e)
        



            


