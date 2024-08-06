from model_training_base.dao.storage_factory import StorageFactory
from model_training_base.types.trainer_types import TRAININGSTATUS

class ModelStorage:
    def __init__(self, config, model_storage_dao = None):
        self.__config = config
        self.__model_storage_dao = model_storage_dao or StorageFactory.make_model_storage(self.__config)


    def create_training_session(self):
        return self.__model_storage_dao.create_training_session()
    
    def start_model_training(self):
        latest_model_session = self.__model_storage_dao.get_latest_model_training_execution()

        if latest_model_session.status == TRAININGSTATUS.FINISHED:
            return latest_model_session
        
        self.__model_storage_dao.change_training_model_status(latest_model_session.execution_id, TRAININGSTATUS.INPROGRESS)
        return self.__model_storage_dao.get_model_training_execution(latest_model_session.execution_id)
    
    def save_model(self, execution_id, model_to_be_saved):
        self.__model_storage_dao.save_model(execution_id, model_to_be_saved)

    def get_model_training_execution(self, execution_id):
        return self.__model_storage_dao.get_model_training_execution(execution_id)

    def get_latest_trained_model(self):
        return self.__model_storage_dao.get_latest_trained_model()
    
    def get_trained_model_by_execution_id(self, execution_id):
        return self.__model_storage_dao.get_trained_model_by_execution_id(execution_id)
