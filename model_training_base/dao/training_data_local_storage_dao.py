import json
import os
import shutil
import uuid
from model_training_base.dao.training_data_storage_dao import TrainingDataStorageDao
from model_training_base.types.trainer_types import SavedTrainingData

class TrainingDataLocalStorageDao(TrainingDataStorageDao):
    def __init__(self, config):
        self.__config = config
        self.__folder_path = self.__config.storage_url + "/data"
        return
    
    def get_training_data(self, model_key): 
        id = uuid.uuid5(uuid.UUID(self.__config.model_uuid), model_key)
        file_path = self.__folder_path + "/" + str(id) + ".json"

        saved_training_data = {}

        if os.path.exists(file_path):   
            f = open(file_path)
            saved_training_data = json.load(f)
            f.close()

        return SavedTrainingData(saved_training_data)
    
    def save_data(self, model_key, new_data):
        id = uuid.uuid5(uuid.UUID(self.__config.model_uuid), model_key)
        file_path = self.__folder_path + "/" + str(id) + ".json"
        os.makedirs(self.__folder_path, exist_ok=True)

        saved_training_data = {}
        saved_training_data["modelKey"] = model_key
        saved_training_data["data"] = new_data

        f = open(file_path, 'w')
        f.write(json.dumps(saved_training_data))
        f.close()

    def get_all_training_data(self):
        all_saved_training_data = []
        
        if os.path.exists(self.__folder_path):
            files = [f for f in os.listdir(self.__folder_path) if os.path.isfile(self.__folder_path + "/" + f)]
            for file in files:
                f = open(self.__folder_path + "/" + file)
                saved_training_data = json.load(f)
                f.close()
                all_saved_training_data.append(SavedTrainingData(saved_training_data))

        return all_saved_training_data
    
    def delete_all_training_data(self):
        shutil.rmtree(self.__folder_path, ignore_errors=True)

    def delete_selected_training_data(self, character):
        id = uuid.uuid5(uuid.UUID(self.__config.model_uuid), character)
        file_path = self.__folder_path + "/" + str(id) + ".json"

        if os.path.exists(file_path):
            os.remove(file_path)