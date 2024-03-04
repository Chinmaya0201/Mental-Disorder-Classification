import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:

    trained_model_file_path : str = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_trainer(self, train_array, test_array):

        try:

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                        'Decision Tree': DecisionTreeClassifier(),
                        'Random Forest': RandomForestClassifier(),
                        'KNN': KNeighborsClassifier()
                    }
            
            params={
                "Decision Tree": {
                    'criterion':["gini", "entropy"],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':["gini", "entropy"],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNN" :{}
                
            }

            model_report : dict = evaluate_models(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= models, param= params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)

            return acc_score, best_model_name

        except Exception as e:

            logging.info(e)
            raise CustomException(e)