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

@dataclass
class DataTransformationConfig():

    preprocessor_object_file_path : str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,path:str):

        

