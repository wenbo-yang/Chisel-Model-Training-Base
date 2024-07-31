from model_training_base.dao.model_storage_dao import ModelStorageDao

class ModelDocumentDBStorageDao(ModelStorageDao):
    def __init__(self, config):
        self.__config = config
        return