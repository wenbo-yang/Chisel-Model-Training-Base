from enum import Enum

class TRAININGSTATUS(Enum):
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