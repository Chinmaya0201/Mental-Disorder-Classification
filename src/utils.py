import os
import dill

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

if __name__ == "__main__":

    pass