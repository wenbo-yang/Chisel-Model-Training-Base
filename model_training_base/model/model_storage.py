from model_training_base.dao.storage_factory import StorageFactory

class ModelStorage:
    def __init__(self, config, model_storage_dao):
        self.__config = config
        self.__model_storage_dao = model_storage_dao or StorageFactory.make_model_storage(config)

    def save_data(self, training_data): 
        pass

    def get_all_training_data(self):
        pass
