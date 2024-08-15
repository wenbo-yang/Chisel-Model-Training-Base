import sys
sys.path.append("./")
sys.path.append("./model_training_base")

import pytest

from commonLib import load_and_compress_neural_net_training_images
from model_training_base.controller.model_training_base_controller import ModelTrainingBaseController
from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.model.model_storage import ModelStorage
from model_training_base.model.training_data_storage import TrainingDataStorage
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.types.trainer_types import COMPRESSIONTYPE, TRAININGDATATYPE, TRAININGSTATUS, ReceivedTrainingData
from uuid import uuid4

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = uuid4()
config.temp_image_path = "./dev/tempImage"
config.enough_accuracy_epoch_count = 1
config.loss_threshold = 0.2
config.accuracy_threshold = 0.2

model_local_storage_dao = ModelLocalStorageDao(config)
training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
model_storage = ModelStorage(config)
training_data_storage = TrainingDataStorage(config)

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    model_local_storage_dao.delete_all_training_executions()
    training_data_local_storage_dao.delete_all_training_data()

class ModelTrainingTestController(ModelTrainingBaseController):
    def __init__(self, config, model_training_model = None, background_tasks = None):
        super(ModelTrainingTestController, self).__init__(config, model_training_model, background_tasks)

    def upload_training_data(self, received_training_data):
        return super()._upload_training_data(received_training_data)

def test_create_controller_should_create_a_controller_object():
    model_training_test_controller = ModelTrainingTestController(config)
    assert model_training_test_controller != None

def test_upload_training_data_should_return_created_status():
    model_training_test_controller = ModelTrainingTestController(config)
    training_data = load_and_compress_neural_net_training_images()
    
    for td in training_data:
        td["dataType"] = TRAININGDATATYPE.PNG
        td["compression"] = COMPRESSIONTYPE.GZIP
        assert model_training_test_controller.upload_training_data(ReceivedTrainingData(td)) == TRAININGSTATUS.CREATED
    
