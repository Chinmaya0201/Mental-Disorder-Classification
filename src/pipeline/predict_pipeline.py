import os 
import sys 

import numpy as np
import pandas as pd 

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object 


class PredictPipeline():

    def __init__(self):

        pass

    def predict(self, features):

        try:
            
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            label_encoder_path = os.path.join('artifacts','label_encoder.pkl')

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path= label_encoder_path)

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            # print('predict -------------------------------------',preds)
            # ('np.array(int(preds[0])).reshape(1) ------------ ', np.array(int(preds[0])).reshape(1))
            # get the predict value using index, convert it into integer, then into a array, then reshape into 1D array
            result = label_encoder.inverse_transform(np.array(int(preds[0])).reshape(1))
            #result = label_encoder.inverse_transform(np.array(preds[0]).reshape(1))
            return result
        
        except Exception as e:

            logging.info(e)
            raise CustomException(e)
        
class CustomData():

    def __init__(
            self,
            sadness:str,
            euphoric:str,
            exhausted:str,
            sleep_dissorder:str,
            mood_swing,
            suicidal_thoughts,
            anorxia,
            authority_respect,
            try_explanation,
            aggressive_response,
            ignore_move_on,
            nervous_break_down,
            admit_mistakes,
            overthinking,
            sexual_activity,
            concentration,
            optimisim

    ):
        
        self.sadness = sadness
        self.euphoric = euphoric
        self.exhausted = exhausted
        self.sleep_dissorder =sleep_dissorder

        if mood_swing == "YES":
            self.mood_swing = 1
        else:
            self.mood_swing = 0

        if suicidal_thoughts == "YES":
            self.suicidal_thoughts = 1
        else:
            self.suicidal_thoughts = 0

        if anorxia == "YES":
            self.anorxia = 1
        else:
            self.anorxia = 0

        if authority_respect == "YES":
            self.authority_respect = 1
        else:
            self.authority_respect = 0

        if try_explanation == "YES":
            self.try_explanation = 1
        else:
            self.try_explanation = 0

        if aggressive_response == "YES":
            self.aggressive_response = 1
        else:
            self.aggressive_response = 0

        if ignore_move_on == "YES":
            self.ignore_move_on = 1
        else:
            self.ignore_move_on = 0

        if nervous_break_down == "YES":
            self.nervous_break_down = 1
        else:
            self.nervous_break_down = 0

        if admit_mistakes == "YES":
            self.admit_mistakes = 1
        else:
            self.admit_mistakes = 0

        if overthinking == "YES":
            self.overthinking = 1
        else:
            self.overthinking = 0

        self.sexual_activity = sexual_activity
        self.concentration = concentration
        self.optimisim = optimisim


    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {"Sadness":[self.sadness],
                        "Euphoric":[self.euphoric],
                        "Exhausted":[self.exhausted],
                        "Sleep dissorder":[self.sleep_dissorder],
                        "Mood Swing":[self.mood_swing],
                        "Suicidal thoughts":[self.suicidal_thoughts],
                        "Anorxia":[self.anorxia],
                        "Authority Respect":[self.authority_respect],
                        "Try-Explanation":[self.try_explanation],
                        "Aggressive Response":[self.aggressive_response],
                        "Ignore & Move-On":[self.ignore_move_on],
                        "Nervous Break-down":[self.nervous_break_down],
                        "Admit Mistakes":[self.admit_mistakes],
                        "Overthinking":[self.overthinking],
                        "Sexual Activity":[self.sexual_activity],
                        "Concentration":[self.concentration],
                        "Optimisim":[self.optimisim]}
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:

            logging.info(e)
            raise CustomException(e)


        

        