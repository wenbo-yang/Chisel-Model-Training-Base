from enum import Enum

class TRAININGSTATUS(str, Enum):
    CREATED = 'CREATED'
    INPROGRESS = 'INPROGRESS'
    FINISHED = 'FINISHED'
    VOIDED = 'VOIDED'
    NOCHANGE = 'NOCHANGE'
    UNKNOWN = 'UNKNOWN'

class ModelTrainingExecution:
    def __init__(self, model_training_execution_json = None):
        self.__model_training_execution_json = model_training_execution_json or dict()
        return
    
    @property
    def execution_id(self):
        return self.__model_training_execution_json["executionId"] if "executionId" in self.__model_training_execution_json else ""
    
    @execution_id.setter
    def execution_id(self, execution_id):
        self.__model_training_execution_json["executionId"] = execution_id

    @property
    def updated(self):
        return self.__model_training_execution_json["updated"] if "updated" in self.__model_training_execution_json else 0
    
    @updated.setter
    def updated(self, updated):
        self.__model_training_execution_json["updated"] = updated

    @property
    def status(self):
        return self.__model_training_execution_json["status"] if "status" in self.__model_training_execution_json else TRAININGSTATUS.UNKNOWN
    
    @status.setter
    def status(self, status):
        self.__model_training_execution_json["status"] = status
    
    @property 
    def model_path(self):
        return self.__model_training_execution_json["modelPath"] if "modelPath" in self.__model_training_execution_json else ""

    @model_path.setter
    def model_path(self, model_path):
        self.__model_training_execution_json["modelPath"] = model_path

    def json(self): 
        return self.__model_training_execution_json
    
class SavedTrainingData:
    def __init__(self, saved_training_data = None):
        self.__saved_training_data = saved_training_data or {}
        self.__data_map = dict()
        self.__parse_data_into_map()

    @property
    def model_key(self):
        return self.__saved_training_data["modelKey"] if "modelKey" in self.__saved_training_data else ""
    
    @model_key.setter
    def model_key(self, model_key):
        self.__saved_training_data["modelKey"] = model_key

    @property
    def data(self):
        return self.__data_map
    
    def size(self):
        return len(self.__data_map)

    def has(self, key): 
        return key in self.__data_map
    
    def set(self, key, value):
        self.__data_map[key] = value
    
    def json(self): 
        self.__saved_training_data["data"] = self.__data_map
        return self.__saved_training_data
    
    def __parse_data_into_map(self): 
        if "data" in self.__saved_training_data:
            self.__data_map = self.__saved_training_data["data"]

    
    
