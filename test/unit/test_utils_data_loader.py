import sys
sys.path.append("./")
sys.path.append("./model_training_base")

import gzip
import base64
import pytest

from genericpath import isfile
from os import listdir
from uuid import uuid4
from model_training_base.dao.model_local_storage_dao import ModelLocalStorageDao
from model_training_base.model.model_training_model import ModelTrainingModel
from model_training_base.types.config import ModelTrainingBaseConfig
from model_training_base.dao.training_data_local_storage_dao import TrainingDataLocalStorageDao
from model_training_base.utils.data_loader import DataLoader
from model_training_base.utils.data_piper import DataPiper

config = ModelTrainingBaseConfig()
config.storage_url = "./dev/localStorage"
config.env = "development"
config.model_uuid = uuid4()
config.temp_image_path = "./dev/tempImage"
config.data_size = 50

training_data_local_storage_dao = TrainingDataLocalStorageDao(config)
data_piper = DataPiper(config)
model_training_model = ModelTrainingModel(config)
model_local_storage_dao = ModelLocalStorageDao(config)

def __load_and_compress_test_image(test_image_location):
    encoded_string = ""
    with open(test_image_location, "rb") as image_file:   
        compressed_bytes = gzip.compress(image_file.read())
        encoded_string = str(base64.b64encode(compressed_bytes), "ascii")
    return encoded_string

def _load_and_compress_test_image1():
    test_image_location = "./test/unit/test_data/zou_character_skeleton.png"
    return __load_and_compress_test_image(test_image_location)

def _load_and_compress_test_image2():
    test_image_location = "./test/unit/test_data/zou_human_skeleton.png"
    return __load_and_compress_test_image(test_image_location)

@pytest.fixture(autouse=True)
def run_after_tests():
    yield
    training_data_local_storage_dao.delete_all_training_data()
    model_local_storage_dao.delete_all_training_executions()
    data_piper.delete_temp_images()

def test_putting_data_to_temp_image_folder_data_loader_should_load_train_loader_and_test_loader():
    compressed_data1 = _load_and_compress_test_image1()
    compressed_data2 = _load_and_compress_test_image2()

    model_training_model.store_training_data("zou", [compressed_data1, compressed_data2])
    saved_training_data = training_data_local_storage_dao.get_all_training_data()
    data_piper.unzip_data(saved_training_data)
    image_dir = config.temp_image_path + "/" + "zou"
    files = [f for f in listdir(image_dir) if isfile(image_dir + "/" + f)]
    assert len(files) == 2

    data_loader = DataLoader(config)
    data_loader.load_data()
    assert data_loader.train_loader != None
    assert data_loader.test_loader != None

def test_not_putting_data_to_temp_image_folder_data_loader_should_not_return_train_loader_and_test_loader():
    data_loader = DataLoader(config)
    data_loader.load_data()
    assert data_loader.train_loader == {}
    assert data_loader.test_loader == {}
