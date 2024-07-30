from abc import ABC, abstractmethod

class ModelStorageDao(ABC):
    @abstractmethod
    def get_latest_model(self):
        pass

    @abstractmethod
    def get_latest_model_by_status(self, status): 
        pass

    @abstractmethod
    def createTrainingSession(self): 
        pass
    
    @abstractmethod
    def delete_all_training_executions(self): 
        pass

    @abstractmethod
    def delete_selected_training_execution(self, execution_id):
        pass

    @abstractmethod
    def change_training_model_status(self, execution_id, status): 
        pass

    @abstractmethod
    def save_model(self, execution_id, model_to_be_saved):
        pass

    @abstractmethod
    def get_model_training_execution(self, execution_id): 
        pass

    @abstractmethod
    def get_latest_trained_model(self):
        pass

    @abstractmethod
    def get_trained_model_by_execution_id(self, execution_id): 
        pass