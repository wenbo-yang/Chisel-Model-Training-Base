import sys
sys.path.append("./")
sys.path.append("./model_training_base")

from uuid import uuid4
import pytest
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.model.training_data_storage import TrainingDataStorage

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = uuid4()

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    training_data_local_storage_dao.delete_all_training_data()

def test_create_data_storage_model_get_all_training_data_from_should_return_nothing():
    train_storage_model = TrainingDataStorage(config)
    assert train_storage_model != None

    all_data = train_storage_model.get_all_training_data()
    assert len(all_data) == 0

def test_should_be_able_to_get_saved_training_data():
    train_storage_model = TrainingDataStorage(config)
    data = {}
    data["key1"] =  "value1"
    train_storage_model.save_data("test", data)
    
    all_data = train_storage_model.get_all_training_data()

    assert len(all_data) == 1
    assert len(all_data[0].data) == 1
    assert all_data[0].model_key == "test"
    assert all_data[0].has("key1")

def test_save_multiple_data_with_multiple_chars_should_be_able_to_get_saved_training_data():
    train_storage_model = TrainingDataStorage(config)
    train_storage_model.save_data("test1", {"key11": "value11"})
    train_storage_model.save_data("test1", {"key12": "value12"})

    train_storage_model.save_data("test2", {"key21": "value21"})
    train_storage_model.save_data("test2", {"key22": "value22"})
    
    all_data = train_storage_model.get_all_training_data()

    test1 = [x for x in all_data if x.model_key == "test1"]
    test2 = [x for x in all_data if x.model_key == "test2"]

    assert len(all_data) == 2
    assert len(test1) == 1
    assert len(test2) == 1
    assert test1[0].data["key11"] == "value11"
    assert test1[0].data["key12"] == "value12"
    assert test2[0].data["key21"] == "value21"
    assert test2[0].data["key22"] == "value22"
    


