
from model_training_base.utils.simple_conv_neural_net import SimpleCNN

class NeuralNetFactory(object):
    @staticmethod 
    def make_neural_network(config, output_size):
        return SimpleCNN(config, output_size)