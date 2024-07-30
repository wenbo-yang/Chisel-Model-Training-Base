from genericpath import isfile
import json
import os
from dao.model_storage_dao import ModelStorageDao

class ModelLocalStorageDao(ModelStorageDao):
    def __init__(self, config):
        self.__config = config
        self.__folder_path = os.path.join(self.__config.storage_url, '/model')
        return
    
    async def get_latest_model(self):
        executions = await self.__get_executions()

        return
    

    async def __get_executions(self):
        files = [f for f in os.listdir(self.__folder_path) if isfile(os.path.join(self.__folder_path, f))]

        for f in files:
            execution = ModelTrainingExecution(json.load(f))

        return