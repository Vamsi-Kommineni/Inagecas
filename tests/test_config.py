from __future__ import annotations

from inagecas_shared.config import Settings


def test_api_keys_parsed_from_csv():
    settings = Settings(api_keys="a, b ,, c")
    assert settings.api_key_set == {"a", "b", "c"}


def test_cors_origins_parsed_from_csv():
    settings = Settings(cors_origins="http://a.com, http://b.com")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_empty_auth_is_empty_set():
    assert Settings(api_keys="").api_key_set == set()


def test_is_production_flag():
    assert Settings(environment="production").is_production
    assert not Settings(environment="development").is_production
