from model.model_storage import ModelStorage
from model.training_data_storage import TrainingDataStorage


class ModelTrainingModel:
    def __init__(self, config, model_storage, training_data_storage):
        self.__config = config
        self.__model_storage = model_storage or ModelStorage(config)
        self.__training_data_storage = training_data_storage or TrainingDataStorage(config)

    async def store_training_data(self, model, uncompressed_data, compressed_data):
        pass

    async def start_model_training(self):
        pass

    async def train_model(self):
        pass

    async def get_model_training_execution(self, execution_id):
        pass

    async def get_last_training_model(self):
        pass

    async def get_trained_model_by_execution_id(self, execution_id):
        pass

    async def __process_saved_data(self, data):
        pass
