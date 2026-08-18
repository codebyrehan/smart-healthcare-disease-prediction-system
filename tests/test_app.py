import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'


def test_prediction_rejects_non_json(client):
    response = client.post('/api/predict', data='not-json', content_type='text/plain')
    assert response.status_code in (400, 503)


def test_metadata_endpoint_is_safe_when_dataset_unavailable(client):
    response = client.get('/api/metadata')
    assert response.status_code in (200, 503)
