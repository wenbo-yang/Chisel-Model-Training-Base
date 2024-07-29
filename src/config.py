import json

class CharacterModelTrainingServiceConfig: 
    def __init__(self):
        f = open("./Chisel-Global-Service-Configs/configs/globalServicePortMappings.json")
        self.global_service_port_mappings = json.load(f)
        f.close()

        f = open("./configs/service.config.json")
        self.service_config = json.load(f)
        f.close()

        self.env = "development"
        self.service_name = self.service_config["serviceName"]
        self.short_name = self.service_config["shortName"]
        self.storage = [x for x in self.service_config["storage"] if x["env"] == self.env][0]
        self.service_address = [x for x in self.service_config["serviceAddress"] if x["env"] == self.env][0]["url"]
        self.service_port_http = self.global_service_port_mappings[self.service_name][self.env]["http"]
        self.service_port_https = self.global_service_port_mappings[self.service_name][self.env]["https"]
