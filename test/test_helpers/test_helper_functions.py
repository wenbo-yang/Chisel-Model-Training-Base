
import base64
from genericpath import isdir, isfile
import gzip
import os

def load_and_compress_test_image(test_image_location):
    encoded_string = ""
    with open(test_image_location, "rb") as image_file:   
        compressed_bytes = gzip.compress(image_file.read())
        encoded_string = str(base64.b64encode(compressed_bytes), "ascii")
    return encoded_string

def load_and_compress_neural_net_training_images(root_folder):
    directories = [d for d in os.listdir(root_folder) if isdir(root_folder + "/" + d)]
    training_data = []
    for d in directories:
        sub_dir = root_folder + "/" + d;
        files = [f for f in os.listdir(sub_dir) if isfile(sub_dir + "/" + f)]
        compressed_data_array = []
        for f in files:
            file_path = sub_dir + "/" + f
            compressed_data_array.append(load_and_compress_test_image(file_path))
       
        training_data.append({"modelKey": d, "data": compressed_data_array})
    
    return training_data

