import sys
sys.path.append("./")
sys.path.append("./model_training_base")
sys.path.append("./test")

import pytest
import uuid

from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.model.model_training_model import ModelTrainingModel
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.utils.data_loader import DataLoader
from model_training_base.utils.data_piper import DataPiper
from model_training_base.utils.neural_net_trainer import NeuralNetTrainer
from test_helpers.test_helper_functions import load_and_compress_neural_net_training_images, load_and_compress_test_image

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = str(uuid.uuid4())
config.temp_image_path = "./dev/tempImage"
config.enough_accuracy_epoch_count = 1
config.loss_threshold = 0.2
config.accuracy_threshold = 0.2

training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
data_piper = DataPiper(config)
model_training_model = ModelTrainingModel(config)
model_local_storage_dao = ModelLocalStorageDao(config)

def _load_and_compress_test_image1():
    test_image_location = "./test/unit/test_data/zou_character_skeleton.png"
    return load_and_compress_test_image(test_image_location)

def _load_and_compress_test_image2():
    test_image_location = "./test/unit/test_data/zou_human_skeleton.png"
    return load_and_compress_test_image(test_image_location)

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    training_data_local_storage_dao.delete_all_training_data()
    model_local_storage_dao.delete_all_training_executions()
    data_piper.delete_temp_images()

def test_create_neural_net_trainer_should_create_an_instance_of_that_class():
    neural_net_trainer = NeuralNetTrainer(config)
    assert neural_net_trainer != None

def test_saving_and_then_get_all_training_data_then_feed_into_neural_net_trainer_should_load_data_to_neural_net():
    compressed_data1 = _load_and_compress_test_image1()
    compressed_data2 = _load_and_compress_test_image2()

    model_training_model.store_training_data("zou", [compressed_data1, compressed_data2])
    saved_training_data = training_data_local_storage_dao.get_all_training_data()
    neural_net_trainer = NeuralNetTrainer(config)
    try:
        neural_net_trainer.load_training_data(saved_training_data)
    except Exception:
        assert False

def test_not_having_training_data_should_raise_exception():
    saved_training_data = training_data_local_storage_dao.get_all_training_data()
    neural_net_trainer = NeuralNetTrainer(config)
    exception_raised = False
    try:
        neural_net_trainer.load_training_data(saved_training_data)
    except Exception:
        exception_raised = True

    assert exception_raised

def test_load_enough_test_to_run_simple_cnn_net_should_pass():
    training_data = load_and_compress_neural_net_training_images("./test/unit/test_data")
    for item in training_data:
        model_training_model.store_training_data(item["modelKey"], item["data"])

    saved_training_data = training_data_local_storage_dao.get_all_training_data()
    assert len(saved_training_data) == 3
    assert len(saved_training_data[0].data) > 0
    assert len(saved_training_data[1].data) > 0
    assert len(saved_training_data[2].data) > 0 

    try:
        neural_net_trainer = NeuralNetTrainer(config)
        neural_net_trainer.load_training_data(saved_training_data)
    except Exception:
        assert False

def test_load_enough_test_to_train_simple_cnn_net_should_pass():
    training_data = load_and_compress_neural_net_training_images("./test/unit/test_data")
    for item in training_data:
        model_training_model.store_training_data(item["modelKey"], item["data"])
    
    saved_training_data = training_data_local_storage_dao.get_all_training_data()
    neural_net_trainer = NeuralNetTrainer(config)
    neural_net_trainer.load_training_data(saved_training_data)
    neural_net_trainer.train()

    assert neural_net_trainer.neural_net_model != None


