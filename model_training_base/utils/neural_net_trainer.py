
import torch
import torch.nn as nn
import torch.optim as optim

from model_training_base.utils.data_loader import DataLoader
from model_training_base.utils.data_piper import DataPiper
from model_training_base.utils.neural_net_factory import NeuralNetFactory

class NeuralNetTrainer:
    def __init__(self, config, data_piper = None, data_loader = None, net = None, torch_interface = None):
        self.__config = config
        self.__data_piper = data_piper or DataPiper(self.__config)
        self.__data_loader = data_piper or DataLoader(self.__config)
        self.__net = net or NeuralNetFactory.make_neural_network(self.__config)
        self.__torch = torch_interface or torch
        
    def load_training_data(self, saved_training_data):
        self.__data_piper.unzip_data(saved_training_data)
        self.__data_loader.load_data()
        if self.__data_loader.test_loader == {} or self.__data_loader.train_loader == {}:
            raise Exception("no test loader and train loader, cannot continue")
            
    def train(self):
        epoch = 0
        train_loss = 0
        valid_loss = 0
        train_corrects = 0
        valid_corrects = 0
        
        history = []

        device = self.__torch.device(self.__get_device())
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__net.parameters(), lr=0.001)

        while self.__no_improvemet([train_loss, valid_loss, train_corrects, valid_corrects], history):
            for i, (images, labels) in enumerate(self.__data_loader.train_loader, 0):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.__net(images)
                loss = criterion(outputs, labels)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_corrects += (labels == self.__torch.max(outputs, 1).indices).sum().item()

            for i, (images, labels) in enumerate(self.__data_loader.train_loader, 0): 
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.__net(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                valid_corrects += (labels == self.__torch.max(outputs, 1).indices).sum().item()



            break

    def __get_device(self):
        if self.__config.use_gpu and self.__torch.cuda.is_available():
            return "cuda"
        
        return "cpu"