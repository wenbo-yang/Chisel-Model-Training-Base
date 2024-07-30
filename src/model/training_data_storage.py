from dao.storage_factory import StorageFactory

class TrainingDataStorage: 
    def __init__(self, config, training_data_storage_dao):
        self.__config = config
        self.__training_data_storage_dao = training_data_storage_dao or StorageFactory.make_training_data_storage(config)

    