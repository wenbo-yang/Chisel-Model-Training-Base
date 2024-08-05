import sys
sys.path.append("./")
sys.path.append("./model_training_base")

import pytest
import json
import os

from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.types.trainer_types import TRAININGSTATUS

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"

class FakeTorch:
    def save(self, object, path):
        f = open(path, "w")
        f.write(json.dumps(object))
        f.close()

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    model_local_storage_dao = ModelLocalStorageDao(config)
    model_local_storage_dao.delete_all_training_executions()

def test_create_model_local_storage_dao_should_not_return_anythin():    
    model_local_storage_dao = ModelLocalStorageDao(config)
    execution = model_local_storage_dao.get_latest_model_training_execution()
    assert execution == None

def test_create_execution_should_create_execution_with_created_status():
    model_local_storage_dao = ModelLocalStorageDao(config)
    execution = model_local_storage_dao.create_training_session()
    expected_saved_execution = model_local_storage_dao.get_latest_model_training_execution()
    assert execution.execution_id == expected_saved_execution.execution_id
    assert expected_saved_execution.status == TRAININGSTATUS.CREATED

def test_create_execution_should_reuse_existing_created_training_session():
    model_local_storage_dao = ModelLocalStorageDao(config)
    execution1 = model_local_storage_dao.create_training_session()
    execution2 = model_local_storage_dao.create_training_session()
    expected_saved_execution = model_local_storage_dao.get_latest_model_training_execution()
    assert execution1.execution_id == expected_saved_execution.execution_id
    assert execution2.execution_id == expected_saved_execution.execution_id
    assert expected_saved_execution.status == TRAININGSTATUS.CREATED


def test_can_change_the_status_of_existing_training_exection():
    model_local_storage_dao = ModelLocalStorageDao(config)
    model_local_storage_dao.create_training_session()
    expected_saved_execution = model_local_storage_dao.get_latest_model_training_execution()
    assert expected_saved_execution.status == TRAININGSTATUS.CREATED
    model_local_storage_dao.change_training_model_status(expected_saved_execution.execution_id, TRAININGSTATUS.INPROGRESS)
    changed_status_execution = model_local_storage_dao.get_latest_model_training_execution()
    assert changed_status_execution.execution_id == expected_saved_execution.execution_id


def test_can_get_execution_by_execution_id():
    model_local_storage_dao = ModelLocalStorageDao(config)
    created_execution = model_local_storage_dao.create_training_session()
    expected_saved_execution = model_local_storage_dao.get_model_training_execution(created_execution.execution_id)
    assert expected_saved_execution.status == TRAININGSTATUS.CREATED
    assert expected_saved_execution.execution_id == created_execution.execution_id

def test_can_save_an_model_and_that_training_execution_should_become_finished():
    model_local_storage_dao = ModelLocalStorageDao(config, FakeTorch())
    created_execution = model_local_storage_dao.create_training_session()
    model_local_storage_dao.save_model(created_execution.execution_id, {"key": "value"})
    finished_model = model_local_storage_dao.get_model_training_execution(created_execution.execution_id)

    assert finished_model.execution_id == created_execution.execution_id
    assert finished_model.status == TRAININGSTATUS.FINISHED
    assert finished_model.model_path != ""
    
def test_can_get_the_model_file_of_the_latest_model():
    model_local_storage_dao = ModelLocalStorageDao(config, FakeTorch())
    created_execution = model_local_storage_dao.create_training_session()
    model_local_storage_dao.save_model(created_execution.execution_id, {"key": "value"})
    model_file_path = model_local_storage_dao.get_latest_trained_model()

    assert model_file_path != ""

def test_can_delete_a_finished_training_execution_and_should_also_delete_the_model_file():
    model_local_storage_dao = ModelLocalStorageDao(config, FakeTorch())
    created_execution = model_local_storage_dao.create_training_session()
    model_local_storage_dao.save_model(created_execution.execution_id, {"key": "value"})
    model_file_path = model_local_storage_dao.get_latest_trained_model()
    assert model_file_path != ""

    model_local_storage_dao.delete_selected_training_execution(created_execution.execution_id)
    assert not os.path.exists(model_file_path)




    
    