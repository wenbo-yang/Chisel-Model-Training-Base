import sys
sys.path.append("./")
sys.path.append("./test")
sys.path.append("./model_training_base")

import pytest
import json
import uuid

from model_training_base.model.training_data_storage import TrainingDataStorage
from model_training_base.model.model_training_model import ModelTrainingModel
from model_training_base.model.model_storage import ModelStorage
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.types.trainer_types import TRAININGSTATUS
from test_helpers.test_helper_functions import load_and_compress_neural_net_training_images 

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = str(uuid.uuid4())
config.temp_image_path = "./dev/tempImage"
config.enough_accuracy_epoch_count = 1
config.loss_threshold = 0.2
config.accuracy_threshold = 0.2

class FakeTorch:
    def save(self, object, path):
        f = open(path, "w")
        f.write(json.dumps(object))
        f.close()

class FakeNeuralNetTrainer: 
    def load_training_data(self, training_data):
        pass

    def train(self):
        pass

    @property
    def neural_net_model(self):
        return {"model": "data"}

model_local_storage_dao = ModelLocalStorageDao(config, FakeTorch())
training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
neural_net_trainer = FakeNeuralNetTrainer()
model_storage = ModelStorage(config, model_local_storage_dao)
training_data_storage = TrainingDataStorage(config)

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    model_local_storage_dao.delete_all_training_executions()
    training_data_local_storage_dao.delete_all_training_data()

def test_create_model_training_model_should_create_object():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    assert model_training_model != None
    
def test_store_training_data_should_create_a_training_session():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    status = model_training_model.store_training_data("model_key", ["compressed_data1"])

    assert status == TRAININGSTATUS.CREATED

def test_store_same_training_data_should_not_create_a_new_training_session():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    status = model_training_model.store_training_data("model_key", ["compressed_data1"])
    assert status == TRAININGSTATUS.CREATED
    status = model_training_model.store_training_data("model_key", ["compressed_data1"])
    assert status == TRAININGSTATUS.NOCHANGE

def test_store_different_data_multiple_times_should_not_create_new_session():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    model_training_model.store_training_data("model_key", ["compressed_data1"])
    execution1 = model_local_storage_dao.get_latest_model_training_execution()
    model_training_model.store_training_data("model_key", ["compressed_data2"])
    execution2 = model_local_storage_dao.get_latest_model_training_execution()

    assert execution1.execution_id == execution2.execution_id
    assert execution1.status == TRAININGSTATUS.CREATED
    assert execution2.status == TRAININGSTATUS.CREATED

def test_start_model_training_and_train_model_should_return_execution_and_turn_execution_to_finished():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    model_training_model.store_training_data("model_key1", ["compressed_data1"])
    model_training_model.store_training_data("model_key2", ["compressed_data2"])
    execution = model_training_model.start_model_training()
    model_training_model.train_model(execution.execution_id)
    model_path = model_training_model.get_latest_training_model()
    latest_execution = model_training_model.get_model_training_execution(execution.execution_id)

    assert model_path != None
    assert execution.status == TRAININGSTATUS.INPROGRESS
    assert latest_execution.execution_id == execution.execution_id
    assert latest_execution.status == TRAININGSTATUS.FINISHED

def test_start_and_train_model_should_turn_execution_to_finished():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    model_training_model.store_training_data("model_key1", ["compressed_data11", "compressed_data12"])
    model_training_model.store_training_data("model_key2", ["compressed_data21", "compressed_data22"])
    execution = model_training_model.start_model_training()
    model_training_model.train_model(execution.execution_id)
    model_path = model_training_model.get_latest_training_model()
    model_from_execution = model_training_model.get_trained_model_by_execution_id(execution.execution_id)

    assert model_path == model_from_execution

def test_can_have_multiple_models():
    model_training_model = ModelTrainingModel(config, model_storage, training_data_storage, neural_net_trainer)
    model_training_model.store_training_data("model_key1", ["compressed_data11", "compressed_data12"])
    model_training_model.store_training_data("model_key2", ["compressed_data21", "compressed_data22"])
    execution = model_training_model.start_model_training()
    model_training_model.train_model(execution.execution_id)
    model_path = model_training_model.get_trained_model_by_execution_id(execution.execution_id)

    model_training_model.store_training_data("model_key3", ["compressed_data31", "compressed_data32"])
    model_training_model.store_training_data("model_key4", ["compressed_data41", "compressed_data42"])
    another_execution = model_training_model.start_model_training()
    model_training_model.train_model(another_execution.execution_id)
    another_model_path = model_training_model.get_trained_model_by_execution_id(another_execution.execution_id)

    assert model_path != another_model_path

def test_store_data_and_then_train_should_successfully_train_an_model():
    model_training_model = ModelTrainingModel(config)
    training_data = load_and_compress_neural_net_training_images("./test/unit/test_data")
    for item in training_data:
        model_training_model.store_training_data(item["modelKey"], item["data"])

    execution = model_training_model.start_model_training()
    model_training_model.train_model(execution.execution_id)

    model_path = model_training_model.get_latest_training_model()
    another_model_path = model_training_model.get_trained_model_by_execution_id(execution.execution_id)

    assert model_path != None
    assert model_path == another_model_path
