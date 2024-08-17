import uuid
from model_training_base.model.model_storage import ModelStorage
from model_training_base.model.training_data_storage import TrainingDataStorage
from model_training_base.types.trainer_types import TRAININGSTATUS, SavedTrainingData
from model_training_base.utils.neural_net_trainer import NeuralNetTrainer

class ModelTrainingModel:
    def __init__(self, config, model_storage = None, training_data_storage = None, neural_net_trainer = None):
        self.__config = config
        self.__model_storage = model_storage or ModelStorage(self.__config)
        self.__training_data_storage = training_data_storage or TrainingDataStorage(self.__config)
        self.__neural_net_trainer = neural_net_trainer or NeuralNetTrainer(self.__config)

    def store_training_data(self, model_key, compressed_data):
        data = {}
        for cd in compressed_data:
            key = str(uuid.uuid5(uuid.UUID(self.__config.model_uuid), str(cd)))
            data[key] = cd
    
        new_data_saved = self.__training_data_storage.save_data(model_key, data)

        if (new_data_saved): 
            self.__model_storage.create_training_session()
            return TRAININGSTATUS.CREATED
        
        return TRAININGSTATUS.NOCHANGE

    def start_model_training(self):
        return self.__model_storage.start_model_training()

    def train_model(self, execution_id):
        all_training_data = self.__training_data_storage.get_all_training_data()

        self.__neural_net_trainer.load_training_data(all_training_data)
        self.__neural_net_trainer.train()
        model_to_be_saved = self.__neural_net_trainer.neural_net_model

        self.__model_storage.save_model(execution_id, model_to_be_saved)
        
    def get_model_training_execution(self, execution_id):
        return self.__model_storage.get_model_training_execution(execution_id)

    def get_latest_training_model(self):
        return self.__model_storage.get_latest_trained_model()

    def get_trained_model_by_execution_id(self, execution_id):
        return self.__model_storage.get_trained_model_by_execution_id(execution_id)

