import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# loads training data from directory
# apply default transform
class DataLoader:
    def __init__(self, config):
        self.__config = config
        self.__folder_path = self.__config.temp_image_path
        self.__batch_size = self.__config.batch_size or 32
        self.__training_set = {}
        self.__test_set = {}
        self.__train_loader = {}
        self.__test_loader = {}
        self.__load_info_from_folder()
        
    
    def __load_info_from_folder(self):
        if not os.path.exists(self.__folder_path):
            return 

        transform = transforms.Compose([transforms.Resize(48), transforms.Grayscale(), transforms.ToTensor()])
        self.__training_set = ImageFolder(self.__folder_path, transform = transform)
        self.__test_set = ImageFolder(self.__folder_path, transform = transform)
        self.__train_loader = torch.utils.data.DataLoader(self.__training_set,
                                          self.__batch_size,
                                          shuffle=True)
        self.__test_loader = torch.utils.data.DataLoader(self.__test_set,
                                          self.__batch_size,
                                          shuffle=True)
        
    
    def get_train_loader(self):
        return self.__train_loader

    def get_test_loader(self):
        return self.__test_loader


        