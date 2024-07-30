from model.model_storage import ModelStorage
from model.model_training_data_storage import ModelTrainingDataStorage


class ModelTrainingModel:
    def __init__(self, config, model_storage, model_training_data_storage):
        self.config = config
        self.model_storage = model_storage or ModelStorage()
        self.model_training_data_storage = model_training_data_storage or ModelTrainingDataStorage()

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
