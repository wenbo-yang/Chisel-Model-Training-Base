import json
import os
import shutil
from uuid import uuid5
from dao.training_data_storage_dao import TrainingDataStorageDao
from types.trainer_types import SavedTrainingData

class TrainingDataLocalStorageDao(TrainingDataStorageDao):
    def __init__(self, config):
        self.__config = config
        return
    
    def get_current_training_data(self, character): 
        uuid = uuid5(character, self.config.model_uuid)
        file_path = os.path.join(self.config.storage_url, "/data", uuid + ".json")

        if not os.path.exists(file_path):
            return {}

        f = open(file_path)
        saved_training_data = json.load(f)
        f.close()
        
        return SavedTrainingData(saved_training_data)
    
    def save_data(self, character, new_data):
        uuid = uuid5(character, self.config.model_uuid)
        file_path = os.path.join(self.config.storage_url, "/data", uuid + ".json")
        os.makedirs(os.path.join(self.config.storage_url, "/data"), exist_ok=True)

        saved_training_data = {}
        saved_training_data["model"] = character
        saved_training_data["data"] = new_data

        f = open(file_path, 'w')
        f.write(json.dumps(saved_training_data))
        f.close()

    def get_all_training_data(self):
        folder_path = os.path.join(self.config.storage_url, "/data")
        all_saved_training_data = []
        
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            for file in files:
                f = open(file)
                saved_training_data = json.load(f)
                f.close()

                all_saved_training_data.append(SavedTrainingData(saved_training_data))


        return all_saved_training_data
    
    def delete_all_training_data(self):
        folder_path = os.path.join(self.config.storage_url, "/data")
        shutil.rmtree(folder_path)

    def delete_selected_training_data(self, character):
        uuid = uuid5(character, self.config.model_uuid)
        file_path = os.path.join(self.config.storage_url, "/data", uuid + ".json")

        if os.path.exists(file_path):
            os.remove(file_path)