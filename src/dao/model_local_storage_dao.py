from dao.model_storage_dao import ModelStorageDao

class ModelLocalStorageDao(ModelStorageDao):
    def __init__(self, config):
        self.__config = config
        return