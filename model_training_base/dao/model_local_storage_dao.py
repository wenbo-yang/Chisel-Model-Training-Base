from genericpath import isfile
import json
import os
from model_training_base.dao.model_storage_dao import ModelStorageDao
from model_training_base.types.trainer_types import ModelTrainingExecution

class ModelLocalStorageDao(ModelStorageDao):
    def __init__(self, config):
        self.__config = config
        self.__folder_path = os.path.join(self.__config.storage_url, '/model')
        return
    
    def get_latest_model(self):
        executions = self.__get_executions()
        return
    

    def __get_executions(self):
        files = [f for f in os.listdir(self.__folder_path) if isfile(os.path.join(self.__folder_path, f))]

        for f in files:
            execution = ModelTrainingExecution(json.load(f))

        return