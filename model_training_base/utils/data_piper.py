import gzip
import os
import shutil
import base64

# receives training data loaded from storage, 
# unzip them into temp image location 
# this location is a private location for dev, shared location for local and ramdisk for production
# also handles clean up after training
class DataPiper: 
    def __init__(self, config):
        self.__config = config
        self.__folder_path = self.__config.temp_image_path
        self.__model_keys = []
    
    def unzip_data(self, saved_training_data_array):
        self.__unzip_compressed_model_data_to_temp_storage(saved_training_data_array)

    def __unzip_compressed_model_data_to_temp_storage(self, saved_training_data_array):
        for item in saved_training_data_array:
            training_data_path = self.__folder_path + "/" + item.model_key
            os.makedirs(training_data_path, exist_ok=True)
            self.__model_keys.append(item.model_key)

            for key in item.data:
                encoded_compressed_data = item.data[key]
                uncompressed_data = gzip.decompress(base64.b64decode(encoded_compressed_data))

                image_name = str(key) + ".png"
                file_name = training_data_path + "/" + image_name

                if not os.path.exists(file_name): 
                    with open(file_name, 'wb') as f:
                        f.write(uncompressed_data)
            
    def delete_temp_images(self):
        shutil.rmtree(self.__folder_path, ignore_errors=True)

    @property
    def model_keys(self):
        return self.__model_keys
