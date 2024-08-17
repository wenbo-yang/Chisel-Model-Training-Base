import sys

sys.path.append("./")
sys.path.append("./model_training_base")

import pytest
import uuid

from commonLib import load_and_compress_neural_net_training_images
from model_training_base.controller.model_training_base_controller import ModelTrainingBaseController
from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.model.model_storage import ModelStorage
from model_training_base.model.training_data_storage import TrainingDataStorage
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.types.trainer_types import COMPRESSIONTYPE, TRAININGDATATYPE, TRAININGSTATUS, ReceivedTrainingData

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = str(uuid.uuid4())
config.temp_image_path = "./dev/tempImage"
config.enough_accuracy_epoch_count = 1
config.loss_threshold = 0.2
config.accuracy_threshold = 0.2

model_local_storage_dao = ModelLocalStorageDao(config)
training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
model_storage = ModelStorage(config)
training_data_storage = TrainingDataStorage(config)

class FakeBackgroundTasks: 
    def add_task(self, function, arg):
        self.__function = function
        self.__arg = arg

    def run(self):
        self.__function(self.__arg)

class ModelTrainingTestController(ModelTrainingBaseController):
    def __init__(self, config, background_tasks_interface, model_training_model = None):
        super(ModelTrainingTestController, self).__init__(config, background_tasks_interface, model_training_model)

    def upload_training_data(self, received_training_data):
        return super()._upload_training_data(received_training_data)
    
    def start_and_train_model(self):
        return super()._start_and_train_model()
    
    def get_model_training_execution(self, execution_id):
        return super()._get_model_training_execution(execution_id)
    
    def get_latest_trained_model(self):
        return super()._get_latest_trained_model()
    
    def get_trained_model_by_execution_id(self, execution_id):
        return super()._get_trained_model_by_execution_id(execution_id)

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    model_local_storage_dao.delete_all_training_executions()
    training_data_local_storage_dao.delete_all_training_data()
    
def test_create_controller_should_create_a_controller_object():
    model_training_test_controller = ModelTrainingTestController(config, FakeBackgroundTasks())
    assert model_training_test_controller != None

def test_upload_training_data_should_return_created_status():
    model_training_test_controller = ModelTrainingTestController(config, FakeBackgroundTasks())
    training_data = load_and_compress_neural_net_training_images()
    
    for td in training_data:
        td["dataType"] = TRAININGDATATYPE.PNG
        td["compression"] = COMPRESSIONTYPE.GZIP
        assert model_training_test_controller.upload_training_data(ReceivedTrainingData(td)) == TRAININGSTATUS.CREATED

def test_start_and_train_model_should_return_immediately_with_execution_and_inprogress_status():
    test_upload_training_data_should_return_created_status()
    model_training_test_controller = ModelTrainingTestController(config, FakeBackgroundTasks())
    execution = model_training_test_controller.start_and_train_model()
    assert execution.execution_id != None
    assert (not execution.model_path) == True
    assert execution.status == TRAININGSTATUS.INPROGRESS


def test_wait_until_model_training_is_finished_get_model_should_return_model_path():
    test_upload_training_data_should_return_created_status()
    fake_background_tasks = FakeBackgroundTasks()
    model_training_test_controller = ModelTrainingTestController(config, background_tasks_interface=fake_background_tasks)
    execution = model_training_test_controller.start_and_train_model()
    fake_background_tasks.run()
    execution = model_training_test_controller.get_model_training_execution(execution.execution_id)

    assert execution.status == TRAININGSTATUS.FINISHED
    assert execution.model_path 

    model_path = model_training_test_controller.get_latest_trained_model()
    assert model_path
    model_path = model_training_test_controller.get_trained_model_by_execution_id(execution.execution_id)
    assert model_path