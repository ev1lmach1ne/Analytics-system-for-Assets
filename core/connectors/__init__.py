import json
import os

_CONNECTORS_FILE = None

def _ensure_file():
    global _CONNECTORS_FILE
    if _CONNECTORS_FILE:
        return _CONNECTORS_FILE
    from core.config import BASE_DATA
    _CONNECTORS_FILE = os.path.join(os.path.dirname(BASE_DATA), '.connectors.json')
    if not os.path.exists(_CONNECTORS_FILE):
        _save([])
    return _CONNECTORS_FILE

def _save(connectors):
    path = _ensure_file()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(connectors, f, indent=2, ensure_ascii=False)

def load_connectors():
    path = _ensure_file()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        _save([])
        return []

def save_connector(config: dict):
    connectors = load_connectors()
    name = config.get('name', '').strip()
    for i, c in enumerate(connectors):
        if c.get('name') == name:
            connectors[i] = config
            _save(connectors)
            return
    connectors.append(config)
    _save(connectors)

def delete_connector(name: str):
    connectors = load_connectors()
    connectors = [c for c in connectors if c.get('name') != name]
    _save(connectors)

def provider_class_for_type(provider_type: str):
    if provider_type == 'oanda':
        from .oanda_provider import OANDAProvider
        return OANDAProvider
    return None
