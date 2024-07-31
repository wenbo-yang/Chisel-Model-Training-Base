from model_training_base.dao.training_data_storage_dao import TrainingDataStorageDao

class TrainingDataDocumentDBStorageDao(TrainingDataStorageDao):
    def __init__(self, config):
        self.__config = config
        return