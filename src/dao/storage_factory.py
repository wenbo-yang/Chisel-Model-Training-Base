from dao.model_document_db_storage_dao import ModelDocumentDBStorageDao
from dao.model_local_storage_dao import ModelLocalStorageDao
from dao.training_data_document_db_storage_dao import TrainingDataDocumentDBStorageDao
from dao.training_data_local_storage_dao import TrainingDataLocalStoragetDao

class StorageFactory:
    @staticmethod
    def make_model_storage(config):
        if config.env == "production":
            return ModelDocumentDBStorageDao(config)
        return ModelLocalStorageDao(config)

    @staticmethod
    def make_training_data_storage(config):
        if config.env == "production":
            return TrainingDataDocumentDBStorageDao(config)
        return TrainingDataLocalStoragetDao(config)
