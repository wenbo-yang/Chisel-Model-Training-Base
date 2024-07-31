from abc import ABC, abstractmethod

class TrainingDataStorageDao(ABC):
    @abstractmethod
    def get_training_data(self, character): 
        pass
    
    @abstractmethod
    def save_data(self, character, newData):
        pass

    @abstractmethod
    def get_all_training_data(self): 
        pass

    @abstractmethod
    def delete_all_training_data(self):
        pass

    @abstractmethod
    def delete_selected_training_data(self, character):
        pass