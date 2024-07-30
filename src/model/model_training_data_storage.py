from dao.storage_factory import StorageFactory

class ModelTrainingDataStorage: 
    def __init__(self, config, model_training_data_storage_dao):
        self.__config = config
        self.__model_training_data_storage_dao = model_training_data_storage_dao or StorageFactory.make_model_training_data_storage(config)

    