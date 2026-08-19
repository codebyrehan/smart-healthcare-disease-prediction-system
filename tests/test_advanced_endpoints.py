from app import app


def test_explainability_endpoint_is_safe_without_model():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/api/explainability')
        assert response.status_code in (200, 503)
        assert response.is_json


def test_benchmark_endpoint_returns_structured_response():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/api/benchmark')
        assert response.status_code in (200, 500, 503)
        assert response.is_json


def test_metadata_contains_quality_summary_when_dataset_available():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/api/metadata')
        if response.status_code == 200:
            assert 'quality_summary' in response.json
            assert 'missing_values' in response.json['quality_summary']
