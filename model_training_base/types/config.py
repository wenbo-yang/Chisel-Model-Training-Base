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

    @property
    def use_gpu(self):
        return self.__model_training_config_base_json["useGPU"] if "useGPU" in self.__model_training_config_base_json else False
    
    @use_gpu.setter
    def use_gpu(self, use_gpu):
        self.__model_training_config_base_json["useGPU"] = use_gpu

    @property
    def loss_threshold(self):
        return self.__model_training_config_base_json["lossThreshold"] if "lossThreshold" in self.__model_training_config_base_json else 0.0001
    
    @loss_threshold.setter
    def loss_threshold(self, threshold):
        self.__model_training_config_base_json["lossThreshold"] = threshold
    
    @property
    def enough_accuracy_epoch_count(self):
        return self.__model_training_config_base_json["enoughAccuracyEpochCount"] if "enoughAccuracyEpochCount" in self.__model_training_config_base_json else 10
    
    @enough_accuracy_epoch_count.setter
    def enough_accuracy_epoch_count(self, count):
        self.__model_training_config_base_json["enoughAccuracyEpochCount"] = count

    @property
    def accuracy_threshold(self):
        return self.__model_training_config_base_json["accuracyThreshold"] if "accuracyThreshold" in self.__model_training_config_base_json else 1.00000
    
    @accuracy_threshold.setter
    def accuracy_threshold(self, threshold):
        self.__model_training_config_base_json["accuracyThreshold"] = threshold

    def json(self):
        return self.__model_training_config_base_json