
from model_training_base.utils import SimpleCNN

class NeuralNetFactory(object):
    @staticmethod 
    def make_neural_network(config):
        return SimpleCNN(config)