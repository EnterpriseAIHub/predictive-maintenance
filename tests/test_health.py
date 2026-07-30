def test_liveness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# /health/ready is not covered here — it requires a real database
# connection, which makes it an integration test, not a unit test.
# Added alongside the integration test suite once persistence exists
# (EDD §21).
