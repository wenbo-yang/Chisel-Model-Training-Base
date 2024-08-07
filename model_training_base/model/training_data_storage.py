from dao.storage_factory import StorageFactory

class TrainingDataStorage: 
    def __init__(self, config, training_data_storage_dao = None):
        self.__config = config
        self.__training_data_storage_dao = training_data_storage_dao or StorageFactory.make_training_data_storage(self.__config)

    def save_data(self, model_key, training_data):     
        current_data = self.__training_data_storage_dao.get_training_data(model_key)
        original_size = current_data.size()
        current_data.data.update(training_data)

        if current_data.size() > original_size:
            self.__training_data_storage_dao.save_data(model_key, current_data.data)
            return True
        return False
        
    def get_all_training_data(self):
        return self.__training_data_storage_dao.get_all_training_data()