from dao.training_data_storage_dao import TrainingDataStorageDao

class TrainingDataLocalStoragetDao(TrainingDataStorageDao):
    def __init__(self, config):
        self.__config = config
        return