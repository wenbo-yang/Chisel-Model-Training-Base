from dao.model_training_data_storage_dao import ModelTrainingDataStorageDao

class ModelTrainingDataDocumentDBStorageDao(ModelTrainingDataStorageDao):
    def __init__(self, config):
        self.__config = config
        return