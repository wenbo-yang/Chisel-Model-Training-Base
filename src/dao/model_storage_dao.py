from abc import ABC, abstractmethod

class ModelStorageDao(ABC):
    @abstractmethod
    async def get_latest_model(self):
        pass

    @abstractmethod
    async def get_latest_model_by_status(self, status): 
        pass

    @abstractmethod
    async def createTrainingSession(self): 
        pass
    
    @abstractmethod
    async def delete_all_training_executions(self): 
        pass

    @abstractmethod
    async def delete_selected_training_execution(self, execution_id):
        pass

    @abstractmethod
    async def change_training_model_status(self, execution_id, status): 
        pass

    @abstractmethod
    async def save_model(self, execution_id, model_to_be_saved):
        pass

    @abstractmethod
    async def get_model_training_execution(self, execution_id): 
        pass

    @abstractmethod
    async def get_latest_trained_model(self):
        pass

    @abstractmethod
    async def get_trained_model_by_execution_id(self, execution_id): 
        pass