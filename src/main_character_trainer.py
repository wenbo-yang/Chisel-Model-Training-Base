import ssl
from subprocess import Popen
import sys
import uvicorn
from app import app
from config import CharacterModelTrainingServiceConfig

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("./certs/cert.crt", keyfile="./certs/key.pem")

character_model_training_config = CharacterModelTrainingServiceConfig()

if __name__ == "__main__":
    print(sys.argv)

    if sys.argv and len(sys.argv) > 1 and ("run_http" in sys.argv): 
        uvicorn.run(app, host=character_model_training_config.service_address, port = character_model_training_config.service_port_http)
    else:
        Popen(["python", "src/main_character_trainer.py", "run_http"])
        uvicorn.run(app, host=character_model_training_config.service_address, port = character_model_training_config.service_port_https, ssl_keyfile="./certs/key.pem", ssl_certfile="./certs/cert.crt")
