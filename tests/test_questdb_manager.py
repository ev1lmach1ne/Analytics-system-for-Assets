import os
import socket

import pytest

import core.questdb_manager as qm


def test_urls_windows():
    url = f'{qm._GITHUB_RELEASE}/questdb-{qm.QUESTDB_VERSION}-rt-windows-x86-64.tar.gz'
    assert qm.QUESTDB_VERSION in url
    assert url.startswith('https://github.com/questdb/questdb/releases/download/')
    assert url.endswith('rt-windows-x86-64.tar.gz')


def test_url_mac_generic_package():
    url = f'{qm._GITHUB_RELEASE}/questdb-{qm.QUESTDB_VERSION}-no-jre-bin.tar.gz'
    assert url.endswith('no-jre-bin.tar.gz')


def test_arch_mac_mapea_apple_silicon_e_intel(monkeypatch):
    monkeypatch.setattr(qm.platform, 'machine', lambda: 'arm64')
    assert qm._arch_mac() == 'aarch64'
    monkeypatch.setattr(qm.platform, 'machine', lambda: 'x86_64')
    assert qm._arch_mac() == 'x64'


def test_es_host_local(monkeypatch):
    monkeypatch.setattr(qm, 'QUESTDB_HOST', 'localhost')
    assert qm._es_host_local()
    monkeypatch.setattr(qm, 'QUESTDB_HOST', '127.0.0.1')
    assert qm._es_host_local()
    monkeypatch.setattr(qm, 'QUESTDB_HOST', '192.168.1.50')
    assert not qm._es_host_local()


def test_is_reachable_puerto_cerrado_false_rapido(monkeypatch):
    # puerto libre: pedirle al SO uno, cerrarlo y comprobar que nadie escucha
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    puerto_libre = s.getsockname()[1]
    s.close()
    monkeypatch.setattr(qm, 'QUESTDB_HOST', '127.0.0.1')
    monkeypatch.setattr(qm, 'QUESTDB_PG_PORT', puerto_libre)
    assert qm.is_reachable(timeout=0.5) is False


def test_is_reachable_puerto_abierto_true(monkeypatch):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)
    puerto = srv.getsockname()[1]
    try:
        monkeypatch.setattr(qm, 'QUESTDB_HOST', '127.0.0.1')
        monkeypatch.setattr(qm, 'QUESTDB_PG_PORT', puerto)
        assert qm.is_reachable(timeout=1.0) is True
    finally:
        srv.close()


def test_ensure_running_host_remoto_no_toca_nada(monkeypatch):
    monkeypatch.setattr(qm, 'is_reachable', lambda timeout=1.5: False)
    monkeypatch.setattr(qm, 'QUESTDB_HOST', '10.0.0.5')

    llamado = {'descarga': False, 'arranque': False}
    monkeypatch.setattr(qm, 'descargar_runtime', lambda *a, **k: llamado.__setitem__('descarga', True))
    monkeypatch.setattr(qm, 'arrancar_bundled', lambda *a, **k: llamado.__setitem__('arranque', True))

    ok, msg = qm.ensure_running()
    assert ok is False
    assert not llamado['descarga'] and not llamado['arranque']
    assert 'remoto' in msg or '10.0.0.5' in msg


def test_ensure_running_so_no_soportado(monkeypatch):
    monkeypatch.setattr(qm, 'is_reachable', lambda timeout=1.5: False)
    monkeypatch.setattr(qm, 'QUESTDB_HOST', 'localhost')
    monkeypatch.setattr(qm.platform, 'system', lambda: 'Linux')

    llamado = {'descarga': False}
    monkeypatch.setattr(qm, 'descargar_runtime', lambda *a, **k: llamado.__setitem__('descarga', True))

    ok, msg = qm.ensure_running()
    assert ok is False
    assert not llamado['descarga']
    assert 'no soportad' in msg.lower()


def test_ensure_running_ya_disponible_no_hace_nada(monkeypatch):
    monkeypatch.setattr(qm, 'is_reachable', lambda timeout=1.5: True)
    llamado = {'descarga': False, 'arranque': False}
    monkeypatch.setattr(qm, 'descargar_runtime', lambda *a, **k: llamado.__setitem__('descarga', True))
    monkeypatch.setattr(qm, 'arrancar_bundled', lambda *a, **k: llamado.__setitem__('arranque', True))

    ok, msg = qm.ensure_running()
    assert ok is True
    assert not llamado['descarga'] and not llamado['arranque']


def test_localizar_ejecutable_encuentra_bin_anidado(tmp_path):
    # simula el caso Windows: runtime/bin/java.exe
    (tmp_path / 'bin').mkdir()
    java_exe = tmp_path / 'bin' / 'java.exe'
    java_exe.write_text('')
    assert qm._localizar_ejecutable(str(tmp_path), 'java.exe') == str(java_exe)


def test_localizar_ejecutable_tolera_anidado_temurin_mac(tmp_path):
    # simula Temurin en Mac: jre/jdk-21.x+y-jre/Contents/Home/bin/java
    anidado = tmp_path / 'jdk-21.0.5+11-jre' / 'Contents' / 'Home' / 'bin'
    anidado.mkdir(parents=True)
    java_bin = anidado / 'java'
    java_bin.write_text('')
    assert qm._localizar_ejecutable(str(tmp_path), 'java') == str(java_bin)


def test_localizar_ejecutable_carpeta_inexistente():
    assert qm._localizar_ejecutable('/ruta/que/no/existe/nunca', 'java') is None


def test_localizar_ejecutable_sin_carpeta_bin(tmp_path):
    (tmp_path / 'otra_cosa').mkdir()
    assert qm._localizar_ejecutable(str(tmp_path), 'java') is None


def test_aplanar_subdirectorio_unico(tmp_path):
    interior = tmp_path / 'questdb-9.4.3-no-jre-bin'
    interior.mkdir()
    (interior / 'questdb.sh').write_text('#!/bin/bash')
    (interior / 'env.sh').write_text('export X=1')
    qm._aplanar_subdirectorio_unico(str(tmp_path))
    assert (tmp_path / 'questdb.sh').exists()
    assert (tmp_path / 'env.sh').exists()
    assert not interior.exists()


def test_aplanar_subdirectorio_no_toca_si_hay_varios(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    qm._aplanar_subdirectorio_unico(str(tmp_path))
    assert (tmp_path / 'a').exists() and (tmp_path / 'b').exists()


def test_carpeta_local_fuera_de_base_data(monkeypatch):
    # la carpeta de QuestDB nunca debe depender de BASE_DATA (puede ser una
    # ruta elegida por el usuario, de red, etc. — no sitio para una DB viva)
    ruta = qm._carpeta_local()
    assert 'AnalyticsSystemForAssets' in ruta
    assert 'questdb' in ruta.lower()


def test_is_runtime_instalado_so_desconocido(monkeypatch):
    monkeypatch.setattr(qm.platform, 'system', lambda: 'Linux')
    assert qm.is_runtime_instalado() is False


def test_descargar_throttle_progreso_por_mb(monkeypatch, tmp_path):
    # urlretrieve llama al reporthook por cada bloque (~8 KB); sin el
    # filtro por MB entero se dispararían cientos de actualizaciones para
    # un archivo de pocos MB
    total = 3 * 1024 * 1024   # 3 MB
    tam_bloque = 8192

    def fake_urlretrieve(url, destino, reporthook=None):
        bloques = total // tam_bloque + 1
        for i in range(bloques + 1):
            reporthook(i, tam_bloque, total)

    monkeypatch.setattr(qm.urllib.request, 'urlretrieve', fake_urlretrieve)
    llamadas = []
    qm._descargar('http://x', str(tmp_path / 'f'),
                  lambda msg, pct: llamadas.append((msg, pct)), 'Test')
    # un mensaje por cada valor de MB alcanzado (0, 1, 2, 3), nunca por bloque
    assert 1 < len(llamadas) <= 4
    ultimo_msg, ultimo_pct = llamadas[-1]
    assert ultimo_msg == "Descargando Test… 3 MB / 3 MB"
    assert ultimo_pct == 100
