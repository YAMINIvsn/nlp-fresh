from dart.temporal_resolver import applicable_versions


def test_resolves_single_applicable_version():
    versions = [{
        "version_id": "v1", "effective_from": "2024-01-01", "effective_to": "2024-12-31",
        "jurisdiction": "IN", "entity_scope": "bank"
    }]
    result = applicable_versions(versions, "2024-06-01", "IN", "bank")
    assert result["status"] == "resolved"
    assert result["versions"][0]["version_id"] == "v1"


def test_unknown_effective_date_never_resolves():
    versions = [{
        "version_id": "v1", "effective_from": "unknown", "effective_to": "",
        "jurisdiction": "IN", "entity_scope": "bank"
    }]
    result = applicable_versions(versions, "2024-06-01", "IN", "bank")
    assert result["status"] == "review"
