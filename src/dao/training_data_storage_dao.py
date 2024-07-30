from abc import ABC, abstractmethod

class TrainingDataStorageDao(ABC):
    @abstractmethod
    async def get_current_training_data(self, character): 
        pass
    
    @abstractmethod
    async def save_data(self, character, newData):
        pass

    @abstractmethod
    async def get_all_training_data(self): 
        pass

    @abstractmethod
    async def delete_all_training_data(self):
        pass

    @abstractmethod
    async def delete_selected_training_data(self, character = None):
        pass