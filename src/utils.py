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

if __name__ == "__main__":

    pass