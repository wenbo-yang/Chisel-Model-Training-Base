
import torch
import torch.nn as nn
import torch.optim as optim

from model_training_base.utils.data_loader import DataLoader
from model_training_base.utils.data_piper import DataPiper
from model_training_base.utils.neural_net_factory import NeuralNetFactory
from model_training_base.utils.logger_factory import LoggerFactory

class NeuralNetTrainer:
    def __init__(self, config, data_piper = None, data_loader = None, net = None, torch_interface = None, logger = None):
        self.__config = config
        self.__data_piper = data_piper or DataPiper(self.__config)
        self.__data_loader = data_loader or DataLoader(self.__config)
        self.__neural_net_model = net or NeuralNetFactory.make_neural_network(self.__config)
        self.__torch = torch_interface or torch
        self.__logger = logger or LoggerFactory.make_logger(self.__config, "NeuralNetTrainer")
        
    def load_training_data(self, saved_training_data):
        self.__data_piper.unzip_data(saved_training_data)
        self.__data_loader.load_data()
        if self.__data_loader.test_loader == {} or self.__data_loader.train_loader == {}:
            raise Exception("no test loader and train loader, cannot continue")
            
    def train(self):
        device = self.__torch.device(self.__get_device())

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__neural_net_model.parameters(), lr=0.001)
        
        epoch_count = 0
        accurate_epoch_count = 0

        while accurate_epoch_count < self.__config.enough_accuracy_epoch_count: 
            train_loss = 0
            valid_loss = 0
            train_corrects = 0
            validation_corrects = 0
            total_validation_count = 0
            total_train_count = 0

            for i, (images, labels) in enumerate(self.__data_loader.train_loader, 0):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.__neural_net_model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_corrects += (labels == torch.max(outputs, 1).indices).sum().item()
                total_train_count += len(images)

            for i, (images, labels) in enumerate(self.__data_loader.test_loader, 0): 
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.__neural_net_model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                validation_corrects += (labels == torch.max(outputs, 1).indices).sum().item()
                total_validation_count += len(images)

            self.__logger.log("Epoch [{}]: Train loss = {:.5f}, Valid loss = {:.5f}, Train Acc = {:.5f}, Valid Acc = {:.5f}".format(epoch_count, train_loss/train_corrects, valid_loss/validation_corrects, train_corrects/total_train_count, validation_corrects/total_validation_count))
            epoch_count += 1

            if train_loss/train_corrects < self.__config.loss_threshold and valid_loss/validation_corrects < self.__config.loss_threshold and train_corrects/total_train_count >= 1.00000 and validation_corrects/total_validation_count:
                accurate_epoch_count += 1

    def __get_device(self):
        if self.__config.use_gpu and self.__torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
    @property
    def neural_net_model(self):
        return self.__neural_net_model