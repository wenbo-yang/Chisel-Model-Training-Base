from time import time
from genericpath import isfile
import json
import os
import shutil
import torch
from uuid import uuid4
from model_training_base.dao.model_storage_dao import ModelStorageDao
from model_training_base.types.trainer_types import TRAININGSTATUS, ModelTrainingExecution

class ModelLocalStorageDao(ModelStorageDao):
    def __init__(self, config, torch_interface = None):
        self.__config = config
        self.__folder_path = self.__config.storage_url + "/model"
        self.__torch = torch_interface or torch
        return
    
    def get_latest_model_training_execution(self):
        executions = self.__get_executions()
        sorted(executions, key = lambda execution: execution.updated, reverse = True)
        return executions[0] if executions else None
    
    def get_latest_model_training_execution_by_status(self, status): 
        executions = self.__get_executions()
        sorted(executions, key = lambda execution: execution.updated, reverse = True)
        executions = [x for x in executions if x.status == status]
        return executions[0] if executions else None
    
    def get_model_training_execution(self, execution_id):
        file_path = self.__folder_path + "/" + str(execution_id) + ".json"
        execution = None
        if os.path.exists(file_path):
            f = open(file_path)
            execution = ModelTrainingExecution(json.load(f))
            f.close()
        return execution    
    
    def create_training_session(self):
        os.makedirs(self.__folder_path, exist_ok=True)
        latest = self.get_latest_model_training_execution_by_status(TRAININGSTATUS.CREATED)
        execution = None
        if latest:
            execution=latest
            execution.updated = int(time())
        else:
            execution = ModelTrainingExecution()
            execution.execution_id = str(uuid4())
            execution.updated = int(time())
            execution.status = TRAININGSTATUS.CREATED

        file_path = self.__folder_path + "/" + str(execution.execution_id) + ".json"
        f = open(file_path, 'w')
        f.write(json.dumps(execution.json()))
        f.close()
        return execution
    
    def __get_executions(self):
        executions = []
        
        if os.path.exists(self.__folder_path): 
            files = [f for f in os.listdir(self.__folder_path) if isfile(self.__folder_path + "/" + f)]
            for file in files:
                file = self.__folder_path + "/" + file
                f = open(file)
                content = json.load(f)
                f.close()
                executions.append(ModelTrainingExecution(content))
    
        return executions
    
    def change_training_model_status(self, execution_id, status):
        execution = self.get_model_training_execution(execution_id)
        execution.status = status
        execution.updated = int(time())
        file_path = self.__folder_path + "/" + str(execution_id) + ".json"
        f = open(file_path, "w")
        f.write(json.dumps(execution.json()))
        f.close()

    def delete_all_training_executions(self):
        shutil.rmtree(self.__folder_path, ignore_errors=True)

    # revisit after piping through model training
    def save_model(self, execution_id, model_to_be_saved):
        execution = self.get_model_training_execution(execution_id)    
        if not execution: 
            return
        
        model_path = self.__folder_path + "/" + str(execution.execution_id) + "_model" + ".pt"
        self.__torch.save(model_to_be_saved, model_path)

        execution.model_path = model_path
        execution.updated = int(time())
        execution.status = TRAININGSTATUS.FINISHED
        execution_file_path = self.__folder_path + "/" + str(execution.execution_id) + ".json"
        f = open(execution_file_path, 'w')
        f.write(json.dumps(execution.json()))
        f.close()

    def get_latest_trained_model(self):
        execution = self.get_latest_model_training_execution_by_status(TRAININGSTATUS.FINISHED)
        if execution and execution.model_path and os.path.exists(execution.model_path):
            return execution.model_path
        return None
    
    def delete_selected_training_execution(self, execution_id):
        execution = self.get_model_training_execution(execution_id)
        if execution:
            model_path = execution.model_path

            if os.path.exists(model_path): 
                os.remove(model_path)

            os.remove(self.__folder_path + "/" + str(execution_id) + ".json")

    def get_trained_model_by_execution_id(self, execution_id):
        execution = self.get_model_training_execution(execution_id)
        if execution:
            return execution.model_path

        return None


        
        