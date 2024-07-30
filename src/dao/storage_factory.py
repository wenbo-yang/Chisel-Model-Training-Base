from dao.model_document_db_storage_dao import ModelDocumentDBStorageDao
from dao.model_local_storage_dao import ModelLocalStorageDao
from dao.model_training_data_document_db_storage_dao import ModelTrainingDataDocumentDBStorageDao
from dao.model_training_data_local_storage_dao import ModelTrainingDataLocalStoragetDao

class StorageFactory:
    @staticmethod
    def make_model_storage(config):
        if config.env == "production":
            return ModelDocumentDBStorageDao(config)
        return ModelLocalStorageDao(config)

    @staticmethod
    def make_model_training_data_storage(config):
        if config.env == "production":
            return ModelTrainingDataDocumentDBStorageDao(config)
        return ModelTrainingDataLocalStoragetDao(config)
