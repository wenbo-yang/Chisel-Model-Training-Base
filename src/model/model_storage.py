from dao.storage_factory import StorageFactory

class ModelStorage:
    def __init__(self, config, model_storage_dao):
        self.__config = config
        self.__model_storage_dao = model_storage_dao or StorageFactory.make_model_storage(config)

    async def save_data(self, training_data): 
        pass

    async def get_all_training_data(self):
        pass
