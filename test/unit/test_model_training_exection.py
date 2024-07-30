import json
import sys
sys.path.append("./")

from src.types.trainer_types import TRAININGSTATUS, ModelTrainingExecution

def test_default_model_training_execution_object_should_return_default_values():
    model = ModelTrainingExecution()
    assert model.execution_id == ""
    assert model.model_path == ""
    assert model.status == TRAININGSTATUS.UNKNOWN
    assert model.updated == 0
    assert model.json() == {}
    
