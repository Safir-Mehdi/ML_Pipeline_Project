import os
import sys
import logging
from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
timestemp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
LOG_FILE = f'{timestemp}.log'
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# For debugging
# print('LOG_FILE:', LOG_FILE)  
# print('LOG_FILE_PATH:', LOG_FILE_PATH)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#     print('Scirpt is running')
#     logging.info("***logging has started***")