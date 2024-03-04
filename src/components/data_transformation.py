import os
import sys 

import numpy as np 
import pandas as pd 

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():

    preprocessor_object_file_path : str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):

        try:

            numerical_columns = ["Mood Swing","Suicidal thoughts","Anorxia","Authority Respect",
                                 "Try-Explanation","Aggressive Response","Ignore & Move-On",
                                 "Nervous Break-down","Admit Mistakes","Overthinking","Sexual Activity","Concentration","Optimisim"]
            
            logging.info(f'Numerical Columns defined {numerical_columns}')
            
            categorical_columns = ["Sadness","Euphoric","Exhausted","Sleep dissorder"]

            logging.info(f'Categorical Columns defined {categorical_columns}')

            num_pipeline = Pipeline(
                steps= [
                    ("scaler", StandardScaler(with_mean= False))
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean= False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline, numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            logging.info("preprocessor object defined")

            return preprocessor

        except Exception as e:

            logging.info(e)
            raise CustomException(e)

    def initiate_data_transformation(self,train_path:str, test_path:str):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            drop_columns = ['Patient Number']

            train_df.drop(drop_columns, axis= 1, inplace= True)
            test_df.drop(drop_columns, axis= 1, inplace= True)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Expert Diagnose'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path= self.data_transformation_config.preprocessor_object_file_path,
                obj= preprocessing_obj
            )

            logging.info('Saved preprocessor object')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e)

