from dao.storage_factory import StorageFactory

class TrainingDataStorage: 
    def __init__(self, config, training_data_storage_dao = None):
        self.__config = config
        self.__training_data_storage_dao = training_data_storage_dao or StorageFactory.make_training_data_storage(config)

    def save_data(self, character, training_data):     
        current_data = self.__training_data_storage_dao.get_training_data(character)
        print(current_data)
        original_size = current_data.size()
        current_data.data.update(training_data)

        if current_data.size() > original_size:
            self.__training_data_storage_dao.save_data(character, current_data.data)
        
    def get_all_training_data(self):
        return self.__training_data_storage_dao.get_all_training_data()