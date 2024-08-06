import sys
sys.path.append("./")
sys.path.append("./model_training_base")

import pytest
import json
import os

from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.model.model_storage import ModelStorage
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.types.trainer_types import TRAININGSTATUS

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"

class FakeTorch:
    def save(self, object, path):
        f = open(path, "w")
        f.write(json.dumps(object))
        f.close()

model_local_storage_dao = ModelLocalStorageDao(config, FakeTorch())

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    model_local_storage_dao.delete_all_training_executions()

def test_create_session_should_create_a_session():    
    model_storage = ModelStorage(config, model_local_storage_dao)
    execution = model_storage.create_training_session()
    assert execution != None
    assert execution.status == TRAININGSTATUS.CREATED

def test_create_session_and_start_training_should_return_an_inprogress_session():    
    model_storage = ModelStorage(config, model_local_storage_dao)
    model_storage.create_training_session()
    execution = model_storage.start_model_training()
    assert execution != None
    assert execution.status == TRAININGSTATUS.INPROGRESS

def test_save_model_should_save_model_and_turn_that_execution_to_finished():
    model_storage = ModelStorage(config, model_local_storage_dao)
    model_storage.create_training_session()
    execution = model_storage.start_model_training()

    model_storage.save_model(execution.execution_id, {"key": "value"})
    finished_execution = model_storage.get_model_training_execution(execution.execution_id)

    assert execution.execution_id == finished_execution.execution_id
    assert finished_execution.status == TRAININGSTATUS.FINISHED

def test_get_latest_trained_model_should_return_nothing_if_no_trained_model_exists():
    model_storage = ModelStorage(config, model_local_storage_dao)
    model_storage.create_training_session()
    model_storage.start_model_training()

    model_path = model_storage.get_latest_trained_model()

    assert model_path == None

def test_get_latest_trained_model_should_return_model_if_exists():
    model_storage = ModelStorage(config, model_local_storage_dao)
    model_storage.create_training_session()
    execution = model_storage.start_model_training()

    model_storage.save_model(execution.execution_id, {"key": "value"})
    finished_execution = model_storage.get_model_training_execution(execution.execution_id)

    model_path = model_storage.get_latest_trained_model()

    assert model_path != None
    assert finished_execution.model_path == model_path

def test_get_trained_model_by_execution_id_should_return_model():
    model_storage = ModelStorage(config, model_local_storage_dao)
    model_storage.create_training_session()
    execution = model_storage.start_model_training()
    model_storage.save_model(execution.execution_id, {"key": "value"})
    
    finished_execution = model_storage.get_model_training_execution(execution.execution_id)
    model_path = model_storage.get_trained_model_by_execution_id(finished_execution.execution_id)

    assert model_path != None
    assert finished_execution.model_path == model_path

    