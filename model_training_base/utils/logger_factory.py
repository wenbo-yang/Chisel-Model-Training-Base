
class LocalLogger:
    def __init__(self, config, prefix):
        self.__config = config
        self.__prefix = prefix
    
    def log(self, content):
        print({self.__prefix: content})

class LoggerFactory:
    @staticmethod
    def make_logger(config, prefix):
        return LocalLogger(config, prefix)
