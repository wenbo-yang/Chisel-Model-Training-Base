class ModelTrainingBaseConfig: 
    def __init__(self, model_training_config_base_json = None):
        self.__model_training_config_base_json = model_training_config_base_json or {}

    @property 
    def model_uuid(self):
        return self.__model_training_config_base_json["modelUUID"] if "modelUUID" in self.__model_training_config_base_json else ""
    
    @model_uuid.setter
    def model_uuid(self, uuid):
        self.__model_training_config_base_json["modelUUID"] = uuid

    @property 
    def storage_url(self):
        return self.__model_training_config_base_json["storageUrl"] if "storageUrl" in self.__model_training_config_base_json else ""
    
    @storage_url.setter
    def storage_url(self, url):
        self.__model_training_config_base_json["storageUrl"] = url

    @property
    def temp_image_path(self):
        return self.__model_training_config_base_json["tempImagePath"] if "tempImagePath" in self.__model_training_config_base_json else ""
    
    @temp_image_path.setter
    def temp_image_path(self, temp_image_path): 
        self.__model_training_config_base_json["tempImagePath"] = temp_image_path

    @property
    def batch_size(self):
        return self.__model_training_config_base_json["batchSize"] if "batchSize" in self.__model_training_config_base_json else 0
    
    @batch_size.setter
    def batch_size(self, batch_size): 
        self.__model_training_config_base_json["batchSize"] = batch_size

    @property
    def data_size(self): 
        return self.__model_training_config_base_json["dataSize"] if "dataSize" in self.__model_training_config_base_json else 50
    
    @data_size.setter
    def data_size(self, data_size): 
        self.__model_training_config_base_json["dataSize"] = data_size

    def json(self):
        return self.__model_training_config_base_json