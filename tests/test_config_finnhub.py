import core.config as config


def _reset_cache(monkeypatch, config_path):
    monkeypatch.setattr(config, 'APP_CONFIG_PATH', str(config_path))
    monkeypatch.setattr(config, '_APP_CONFIG_CACHE', {})


def test_finnhub_api_key_por_defecto_vacia(monkeypatch, tmp_path):
    _reset_cache(monkeypatch, tmp_path / 'config.json')
    assert config.get_finnhub_api_key() == ''


def test_set_finnhub_api_key_persiste_y_recarga(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'config.json'
    _reset_cache(monkeypatch, cfg_path)
    monkeypatch.setattr(config, 'BASE_DATA', str(tmp_path))

    config.set_finnhub_api_key('  abc123  ')
    assert config.get_finnhub_api_key() == 'abc123'   # se recorta

    monkeypatch.setattr(config, '_APP_CONFIG_CACHE', {})
    assert config.get_finnhub_api_key() == ''
    config.load_app_config()
    assert config.get_finnhub_api_key() == 'abc123'
