import sys
sys.path.append("./")
sys.path.append("./model_training_base")

from model_training_base.utils.simple_conv_neural_net import SimpleCNN

def test_initiate_simple_conv_neural_net_should_create_object():
    simple_cnn = SimpleCNN({}, 3)
    assert simple_cnn != None
