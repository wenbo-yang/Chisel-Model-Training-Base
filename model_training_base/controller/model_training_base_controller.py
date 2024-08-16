import asyncio
import base64
import gzip
from fastapi import BackgroundTasks
from model_training_base.model.model_training_model import ModelTrainingModel
from model_training_base.types.trainer_types import COMPRESSIONTYPE, TRAININGDATATYPE, ReceivedTrainingData

class ModelTrainingBaseController:
    def __init__(self, config, model_training_model = None, background_tasks = None): 
        self.__config = config
        self.__model_training_model = model_training_model or ModelTrainingModel(config)
        self.__background_tasks = background_tasks or BackgroundTasks()
        return
    
    def _upload_training_data(self, received_training_data: ReceivedTrainingData):
        if received_training_data.data_type != TRAININGDATATYPE.PNG:
            raise Exception("Unsupported Training Data Type")

        if received_training_data.compression != COMPRESSIONTYPE.GZIP:
            self.__compresse_data(received_training_data)

        return self.__model_training_model.store_training_data(received_training_data.model_key, received_training_data.data)

    def _start_and_train_model(self): 
        execution = self.__model_training_model.start_model_training()
        self.__background_tasks.add_task(self.__model_training_model.train_model, execution.execution_id)

        return execution

    def _get_model_training_execution(self, execution_id):
        return self.__model_training_model.get_model_training_execution(execution_id)
        
    def _get_latest_trained_model(self):
        return self.__model_training_model.get_latest_training_model()

    def _get_trained_model_by_execution_id(self, execution_id):
        return self.__model_training_model.get_trained_model_by_execution_id(execution_id)
    
    def _get_background_tasks(self):
        return self.__background_tasks
        
    def __compresse_data(self, received_training_data):
        compressed_data = []
        for d in received_training_data.data:
            compressed_bytes = gzip.compress(d.encode())
            compressed_data.append(str(base64.b64encode(compressed_bytes), "ascii"))

        received_training_data.data = compressed_data
    
    async def __async_train_model(self, execution_id):
        self.__model_training_model.train_model(execution_id)
        