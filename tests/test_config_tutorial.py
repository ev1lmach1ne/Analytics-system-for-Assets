import json

import core.config as config


def _reset_cache(monkeypatch, config_path):
    monkeypatch.setattr(config, 'APP_CONFIG_PATH', str(config_path))
    monkeypatch.setattr(config, '_APP_CONFIG_CACHE', {})


def test_tutorial_visto_por_defecto_falso(monkeypatch, tmp_path):
    _reset_cache(monkeypatch, tmp_path / 'config.json')
    assert config.get_tutorial_visto() is False


def test_set_tutorial_visto_persiste_y_recarga(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'config.json'
    _reset_cache(monkeypatch, cfg_path)
    monkeypatch.setattr(config, 'BASE_DATA', str(tmp_path))

    config.set_tutorial_visto()
    assert config.get_tutorial_visto() is True

    with open(cfg_path, encoding='utf-8') as f:
        data = json.load(f)
    assert data['tutorial_visto'] is True

    monkeypatch.setattr(config, '_APP_CONFIG_CACHE', {})
    assert config.get_tutorial_visto() is False
    config.load_app_config()
    assert config.get_tutorial_visto() is True


def test_load_app_config_sin_clave_tutorial_no_rompe(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'config.json'
    base_dir = tmp_path / 'datos'
    base_dir.mkdir()
    cfg_path.write_text(json.dumps({'base_data': str(base_dir)}), encoding='utf-8')

    _reset_cache(monkeypatch, cfg_path)
    ok = config.load_app_config()
    assert ok is True
    assert config.get_tutorial_visto() is False
