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

    def json(self):
        return self.__model_training_config_base_json