import requests
import pytest
service_url = "https://127.0.0.1:3001"

http_request = requests.Session()
http_request.verify = False

def test_get_health_check(): 
    health_check_url = service_url + "/healthcheck"
    response = http_request.get(health_check_url)

    print(response.json())
    assert response.status_code == 200
    assert response.json() == "I am healthy!!!"


