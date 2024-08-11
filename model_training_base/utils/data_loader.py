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
        self.__train_loader = {}
        self.__test_loader = {}
    
    def load_data(self):
        if not os.path.exists(self.__folder_path):
            return

        transform = transforms.Compose([transforms.Resize(self.__config.data_size), transforms.Grayscale(), transforms.ToTensor()])
        
        training_set = {}
        test_set = {}
        if os.path.exists(self.__folder_path + "/" + "training_set"):
            training_set = ImageFolder(self.__folder_path + "/" + "training_set", transform = transform)
        else:
            training_set = ImageFolder(self.__folder_path, transform = transform)

        if os.path.exists(self.__folder_path + "/" + "test_set"):
            test_set = ImageFolder(self.__folder_path + "/" + "test_set", transform = transform)
        else:
            test_set = ImageFolder(self.__folder_path, transform = transform)

        self.__train_loader = torch.utils.data.DataLoader(training_set,
                                          self.__batch_size,
                                          shuffle=True)
        self.__test_loader = torch.utils.data.DataLoader(test_set,
                                          self.__batch_size,
                                          shuffle=True)
        
    @property
    def train_loader(self):
        return self.__train_loader

    @property
    def test_loader(self):
        return self.__test_loader


        