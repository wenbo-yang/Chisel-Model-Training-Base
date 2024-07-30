from model.model_training_model import ModelTrainingModel

class ModelTrainingController:
    def __init__(self, config, model_training_model): 
        self.__config = config
        self.__model_training_model = model_training_model or ModelTrainingModel(config)
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