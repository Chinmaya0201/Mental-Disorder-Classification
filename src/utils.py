import os
import dill
import pickle

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException

def convert_to_number_string(string):

    try:
        return int(string[0])
    except Exception as e:
        logging.info(e)
        return 0
    
def convert_to_number_yes_no(string):

    try:
        if string == 'YES':
            return 1
        else:
            return 0
        
    except Exception as e:
        logging.info(e)
        return 0
    
def save_object(file_path:str, obj):

    try:
        
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path,"wb") as file_obj:

            dill.dump(obj,file_obj)

    except Exception as e:

        logging.info(e)
        raise CustomException(e)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    try:

        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]

            para = param[list(param.keys())[i]]

            gs = GridSearchCV(model, param_grid= para, cv=3)

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:

        logging.info(e)
        raise CustomException(e)
    
def load_object(file_path):

    try:

        with open(file_path, 'rb') as file_obj:

            return pickle.load(file_obj)

    except Exception as e:

        logging.info(e)
        raise CustomException(e)

if __name__ == "__main__":

    pass