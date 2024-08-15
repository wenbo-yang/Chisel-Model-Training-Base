from fastapi import BackgroundTasks
from model_training_base.model.model_training_model import ModelTrainingModel

class ModelTrainingController:
    def __init__(self, config, model_training_model = None, background_tasks = None): 
        self.__config = config
        self.__model_training_model = model_training_model or ModelTrainingModel(config)
        self.__background_tasks = background_tasks or BackgroundTasks.background_tasks
        return
    
    def _upload_training_data(self, training_data): 
        
        pass

    def _start_model_training(self): 
        pass

    def _train_model(self, execution_id):
        pass

    def _get_model_training_execution(self, execution_id):
        pass

    def _get_latest_trained_model(self, execution_id):
        pass

    def _get_trained_model_by_execution_id(self, execution_id):
        pass

    def __get_decompressed_data(self, training_data):
        pass

    def __get_compressed_data(self, training_data):
        pass