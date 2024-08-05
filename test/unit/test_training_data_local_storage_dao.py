import sys
sys.path.append("./")
sys.path.append("./model_training_base")
import pytest

from uuid import uuid4
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.types.config import ModelTrainingBaseConfig

config = ModelTrainingBaseConfig()
config.model_uuid = uuid4()
config.storage_url = "./dev/localStorage"

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    training_data_local_storage_dao.delete_all_training_data()

def test_create_training_data_local_storage_dao_should_return_no_training_data():    
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    all_training_data = training_data_local_storage_dao.get_all_training_data()
    assert all_training_data == []

def test_get_saved_training_data_should_return_correctly_after_saving_data():
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    training_data_local_storage_dao.save_data("test_character", {"key": "value"})
    all_training_data = training_data_local_storage_dao.get_all_training_data()
    assert len(all_training_data) == 1
    assert all_training_data[0].data["key"] == "value"

def test_get_specific_saved_training_data_should_return_correctly_after_saving_data():
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    training_data_local_storage_dao.save_data("test_character1", {"key1": "value1"})
    training_data_local_storage_dao.save_data("test_character2", {"key2": "value2"})
    specific_data = training_data_local_storage_dao.get_training_data("test_character1")
    assert specific_data.data["key1"] == "value1"

def test_delete_specific_saved_data_should_delete_that_saved_data():
    training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
    training_data_local_storage_dao.save_data("test_character", {"key": "value"})
    training_data_local_storage_dao.delete_selected_training_data("test_character")
    specific_data = training_data_local_storage_dao.get_training_data("test_character")
    all_training_data = training_data_local_storage_dao.get_all_training_data()
    assert specific_data.data == {}
    assert all_training_data == []
   



    
    