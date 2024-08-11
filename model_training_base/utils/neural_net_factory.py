
from model_training_base.utils.simple_conv_neural_net import SimpleCNN

class NeuralNetFactory(object):
    @staticmethod 
    def make_neural_network(config):
        return SimpleCNN(config)