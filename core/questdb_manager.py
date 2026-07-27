"""
core/questdb_manager.py
Descarga, arranca y detiene una QuestDB local "de bolsillo" para que la app
funcione en una máquina nueva sin que la persona instale ni configure nada
(ver core/config.py: DB_CONFIG / QUESTDB_HOST / QUESTDB_PG_PORT).

Solo actúa cuando la conexión configurada apunta a localhost/127.0.0.1: si el
usuario configuró una QuestDB remota a propósito, este módulo nunca la toca
ni descarga nada.

Estrategia por sistema operativo (verificado en la propia máquina de
desarrollo, no es teoría):
- Windows: QuestDB publica un paquete autónomo con Java embebido
  (questdb-<version>-rt-windows-x86-64.tar.gz). Su bin/questdb.exe pasa por
  el Service Control Manager de Windows y pide Administrador incluso para un
  arranque simple; invocando el motor directamente con el Java embebido
  (bin/java.exe -m io.questdb/io.questdb.ServerMain -d <dir>) arranca como
  proceso normal, sin admin.
- macOS: no hay paquete autónomo. Se descarga un JRE portátil de Eclipse
  Temurin (api.adoptium.net) y el paquete genérico de QuestDB sin Java
  (questdb-<version>-no-jre-bin.tar.gz), que trae su propio script oficial
  questdb.sh — se delega el arranque/parada en ESE script (apuntándolo con
  JAVA_HOME al JRE descargado) en vez de reconstruir a mano los flags de JVM
  que usa internamente (-p questdb.jar -m io.questdb/io.questdb.ServerMain
  -d <dir> y varios --add-opens/--enable-native-access): así, si QuestDB
  cambia esos flags en una versión futura, seguimos funcionando igual.

Sin Qt — el envoltorio con progreso en hilo vive en gui/questdb_bootstrap.py.
"""
import json
import os
import platform
import socket
import subprocess
import tarfile
import time
import urllib.request

from core.config import QUESTDB_HOST, QUESTDB_PG_PORT, QUESTDB_HTTP_PORT

QUESTDB_VERSION = '9.4.3'
_GITHUB_RELEASE = f'https://github.com/questdb/questdb/releases/download/{QUESTDB_VERSION}'
_ADOPTIUM_FEATURE = 21   # LTS, compatible con los requisitos de QuestDB

_HOSTS_LOCALES = {'localhost', '127.0.0.1', '::1'}
_SOS_SOPORTADOS = ('Windows', 'Darwin')

_proceso = None   # Popen del proceso que ARRANCAMOS nosotros en Windows (None si no)


# ══════════════ rutas locales ══════════════

def _carpeta_local():
    sistema = platform.system()
    if sistema == 'Windows':
        base = os.getenv('LOCALAPPDATA') or os.path.expanduser('~')
    elif sistema == 'Darwin':
        base = os.path.expanduser('~/Library/Application Support')
    else:
        base = os.path.expanduser('~/.local/share')
    return os.path.join(base, 'AnalyticsSystemForAssets', 'questdb')


def _carpeta_runtime():
    return os.path.join(_carpeta_local(), 'runtime')


def _carpeta_datos():
    return os.path.join(_carpeta_local(), 'data')


def _carpeta_tmp():
    return os.path.join(_carpeta_local(), '_tmp_descarga')


# ══════════════ detección de host/plataforma ══════════════

def _es_host_local():
    return QUESTDB_HOST in _HOSTS_LOCALES


def _arch_mac():
    m = platform.machine().lower()
    return 'aarch64' if m in ('arm64', 'aarch64') else 'x64'


def is_reachable(timeout=1.5):
    """True si algo responde en QUESTDB_HOST:QUESTDB_PG_PORT (ya sea una
    instancia que arrancamos nosotros, que el usuario tenía corriendo, o una
    remota configurada a propósito)."""
    try:
        with socket.create_connection((QUESTDB_HOST, QUESTDB_PG_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _localizar_ejecutable(directorio, nombre):
    """Busca recursivamente <nombre> dentro de una carpeta 'bin' bajo
    `directorio` — evita depender del nombre exacto de la carpeta interna de
    cada release (varía entre versiones; Temurin en Mac anida en
    jdk-.../Contents/Home/bin/java)."""
    if not os.path.isdir(directorio):
        return None
    for raiz, _dirs, ficheros in os.walk(directorio):
        if os.path.basename(raiz) == 'bin' and nombre in ficheros:
            return os.path.join(raiz, nombre)
    return None


def _localizar_java(directorio):
    nombre = 'java.exe' if platform.system() == 'Windows' else 'java'
    return _localizar_ejecutable(directorio, nombre)


# ══════════════ instalado / descarga ══════════════

def is_runtime_instalado():
    sistema = platform.system()
    runtime_dir = _carpeta_runtime()
    if sistema == 'Windows':
        return _localizar_java(runtime_dir) is not None
    if sistema == 'Darwin':
        questdb_sh = os.path.join(runtime_dir, 'questdb', 'questdb.sh')
        jre_ok = _localizar_java(os.path.join(runtime_dir, 'jre')) is not None
        return os.path.isfile(questdb_sh) and jre_ok
    return False


def _descargar(url, destino, progress_cb=None, etiqueta=''):
    """progress_cb(mensaje: str, porcentaje: int|None) — porcentaje None =
    indeterminado (no debería pasar aquí, donde sí conocemos el total)."""
    ultimo_mb = [-1]

    def _reporthook(bloques, tam_bloque, total):
        if not progress_cb or total <= 0:
            return
        descargado = min(bloques * tam_bloque, total)
        descargado_mb = descargado // (1024 * 1024)
        if descargado_mb == ultimo_mb[0]:
            return   # urlretrieve llama esto por cada bloque (~8 KB) — sin
                      # este filtro se dispararían miles de actualizaciones
        ultimo_mb[0] = descargado_mb
        progress_cb(f"Descargando {etiqueta}… {descargado_mb} MB / "
                    f"{total // (1024 * 1024)} MB",
                    int(descargado * 100 / total))
    urllib.request.urlretrieve(url, destino, _reporthook)


def _extraer(tar_path, destino, progress_cb=None, etiqueta=''):
    if progress_cb:
        progress_cb(f"Descomprimiendo {etiqueta}…", None)   # indeterminado:
        # tarfile no expone progreso incremental sin recorrer el índice antes
    os.makedirs(destino, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(destino)


def _url_jre_mac():
    """Resuelve la URL real del último JRE de Temurin para esta arquitectura
    de Mac vía la API pública de Adoptium — evita fijar una versión de
    parche exacta que termine caducando."""
    arch = _arch_mac()
    api = (f'https://api.adoptium.net/v3/assets/latest/{_ADOPTIUM_FEATURE}/hotspot'
          f'?os=mac&architecture={arch}&image_type=jre&vendor=eclipse')
    with urllib.request.urlopen(api, timeout=15) as resp:
        datos = json.load(resp)
    return datos[0]['binary']['package']['link']


def descargar_runtime(progress_cb=None):
    """Descarga y extrae lo necesario para esta plataforma. Solo debe
    llamarse tras confirmar _es_host_local() — nunca contra una QuestDB
    remota configurada a propósito."""
    sistema = platform.system()
    if sistema not in _SOS_SOPORTADOS:
        raise RuntimeError(f"Sistema operativo no soportado: {sistema}")

    runtime_dir = _carpeta_runtime()
    tmp_dir = _carpeta_tmp()
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        if sistema == 'Windows':
            url = f'{_GITHUB_RELEASE}/questdb-{QUESTDB_VERSION}-rt-windows-x86-64.tar.gz'
            tar_path = os.path.join(tmp_dir, 'questdb-windows.tar.gz')
            _descargar(url, tar_path, progress_cb, 'QuestDB')
            _extraer(tar_path, runtime_dir, progress_cb, 'QuestDB')
        elif sistema == 'Darwin':
            url_qdb = f'{_GITHUB_RELEASE}/questdb-{QUESTDB_VERSION}-no-jre-bin.tar.gz'
            tar_qdb = os.path.join(tmp_dir, 'questdb-mac.tar.gz')
            _descargar(url_qdb, tar_qdb, progress_cb, 'QuestDB')
            qdb_dir = os.path.join(runtime_dir, 'questdb')
            _extraer(tar_qdb, qdb_dir, progress_cb, 'QuestDB')
            # el tar trae un único subdirectorio questdb-<version>-no-jre-bin/
            # — aplanarlo para que quede exactamente en qdb_dir/questdb.sh
            _aplanar_subdirectorio_unico(qdb_dir)

            url_jre = _url_jre_mac()
            tar_jre = os.path.join(tmp_dir, 'jre-mac.tar.gz')
            _descargar(url_jre, tar_jre, progress_cb, 'Java (Temurin)')
            _extraer(tar_jre, os.path.join(runtime_dir, 'jre'), progress_cb, 'Java')
            _quitar_cuarentena_mac(runtime_dir)
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _quitar_cuarentena_mac(runtime_dir):
    """Best-effort: `urllib`/`tarfile` no marcan com.apple.quarantine, pero un
    binario descargado y sin firmar puede acabar bloqueado por Gatekeeper de
    todos modos según la versión de macOS. No debe romper la instalación si
    `xattr` no existe o el atributo no está presente."""
    try:
        subprocess.run(['xattr', '-dr', 'com.apple.quarantine', runtime_dir],
                       timeout=15, capture_output=True, text=True)
    except Exception:
        pass


def _aplanar_subdirectorio_unico(carpeta):
    """Si `carpeta` solo contiene un subdirectorio, sube su contenido un
    nivel (los .tar.gz de QuestDB envuelven todo en questdb-<version>-.../)."""
    entradas = os.listdir(carpeta)
    if len(entradas) == 1 and os.path.isdir(os.path.join(carpeta, entradas[0])):
        interior = os.path.join(carpeta, entradas[0])
        for nombre in os.listdir(interior):
            os.rename(os.path.join(interior, nombre), os.path.join(carpeta, nombre))
        os.rmdir(interior)


# ══════════════ arrancar / detener ══════════════

def _env_puertos():
    env = dict(os.environ)
    env['QDB_HTTP_BIND_TO'] = f'0.0.0.0:{QUESTDB_HTTP_PORT}'
    env['QDB_PG_NET_BIND_TO'] = f'0.0.0.0:{QUESTDB_PG_PORT}'
    env['QDB_LINE_TCP_ENABLED'] = 'false'   # protocolo ILP-TCP: este proyecto no lo usa
    return env


def arrancar_bundled():
    sistema = platform.system()
    datos_dir = _carpeta_datos()
    os.makedirs(datos_dir, exist_ok=True)

    if sistema == 'Windows':
        global _proceso
        java_bin = _localizar_java(_carpeta_runtime())
        if not java_bin:
            raise RuntimeError("No se encontró el runtime de QuestDB descargado")
        log_path = os.path.join(_carpeta_local(), 'questdb.log')
        with open(log_path, 'ab') as log_file:
            _proceso = subprocess.Popen(
                [java_bin, '-m', 'io.questdb/io.questdb.ServerMain', '-d', datos_dir],
                stdout=log_file, stderr=subprocess.STDOUT, env=_env_puertos(),
                creationflags=subprocess.CREATE_NO_WINDOW)
        return

    if sistema == 'Darwin':
        qdb_dir = os.path.join(_carpeta_runtime(), 'questdb')
        questdb_sh = os.path.join(qdb_dir, 'questdb.sh')
        java_bin = _localizar_java(os.path.join(_carpeta_runtime(), 'jre'))
        if not os.path.isfile(questdb_sh) or not java_bin:
            raise RuntimeError("No se encontró el runtime de QuestDB descargado")
        os.chmod(questdb_sh, 0o755)
        os.chmod(java_bin, 0o755)   # tarfile suele preservar el bit +x, pero si
                                     # el JRE viene sin él, questdb.sh no puede
                                     # lanzar java y falla de forma opaca
        java_home = os.path.dirname(os.path.dirname(java_bin))   # .../bin/java -> Home
        env = _env_puertos()
        env['JAVA_HOME'] = java_home
        # questdb.sh arranca java en segundo plano y ÉL MISMO vuelve enseguida
        # (gestiona su propio start/status/stop con el PID por su cuenta) —
        # no hay un Popen "nuestro" que trackear, por eso detener_bundled()
        # en Mac llama a `questdb.sh stop` en vez de terminar un proceso.
        resultado = subprocess.run(
            ['/bin/bash', questdb_sh, 'start', '-d', datos_dir],
            env=env, cwd=qdb_dir, timeout=30, capture_output=True, text=True)
        # dejar rastro persistente igual que Windows vuelca a questdb.log
        salida = ((resultado.stdout or '') + (resultado.stderr or '')).strip()
        if salida:
            try:
                with open(os.path.join(_carpeta_local(), 'questdb.log'), 'a',
                          encoding='utf-8') as log_file:
                    log_file.write(f"\n--- questdb.sh start (rc={resultado.returncode}) ---\n")
                    log_file.write(salida + "\n")
            except OSError:
                pass
        # questdb.sh start devolviendo != 0 es un fallo claro (JAVA_HOME mal,
        # permisos, script roto): propagarlo con su salida real en vez de
        # dejar que ensure_running muestre solo "no respondió a tiempo".
        if resultado.returncode != 0:
            detalle = salida or "(sin salida)"
            raise RuntimeError(
                f"questdb.sh start falló (código {resultado.returncode}): {detalle}")
        return

    raise RuntimeError(f"Sistema operativo no soportado: {sistema}")


def detener_bundled():
    """Detiene la instancia que ARRANCAMOS NOSOTROS en esta sesión — nunca
    toca una QuestDB que el usuario ya tuviera corriendo por su cuenta."""
    sistema = platform.system()
    if sistema == 'Windows':
        global _proceso
        if _proceso is None:
            return
        try:
            _proceso.terminate()
            _proceso.wait(timeout=8)
        except Exception:
            try:
                _proceso.kill()
            except Exception:
                pass
        _proceso = None
        return

    if sistema == 'Darwin':
        qdb_dir = os.path.join(_carpeta_runtime(), 'questdb')
        questdb_sh = os.path.join(qdb_dir, 'questdb.sh')
        if os.path.isfile(questdb_sh):
            try:
                subprocess.run(['/bin/bash', questdb_sh, 'stop', '-d', _carpeta_datos()],
                               cwd=qdb_dir, timeout=30, capture_output=True, text=True)
            except Exception:
                pass


def _ultimo_log_questdb(datos_dir, n_lineas=30):
    """QuestDB escribe su propio log bajo <datos_dir>/log/*.log — a diferencia
    de questdb.sh (que solo informa si pudo LANZAR el proceso), este log es
    donde queda el motivo real si el servidor arranca y se cae durante el
    boot (puerto ocupado, JRE incompatible, etc). Devuelve las últimas
    `n_lineas` del fichero más reciente, o None si no hay ninguno."""
    log_dir = os.path.join(datos_dir, 'log')
    if not os.path.isdir(log_dir):
        return None
    try:
        candidatos = [os.path.join(log_dir, f) for f in os.listdir(log_dir)
                     if f.endswith(('.log', '.txt'))]
        if not candidatos:
            return None
        mas_reciente = max(candidatos, key=os.path.getmtime)
        with open(mas_reciente, encoding='utf-8', errors='replace') as fh:
            lineas = fh.readlines()
        return ''.join(lineas[-n_lineas:]).strip() or None
    except OSError:
        return None


# ══════════════ orquestador ══════════════

def ensure_running(progress_cb=None, timeout_arranque=20):
    """Punto de entrada único. Devuelve (ok, mensaje):
    - ya alcanzable (nuestra, del usuario, o remota) -> (True, ...)
    - host remoto configurado a propósito y no responde -> (False, ...) sin tocar nada
    - SO sin estrategia de autoinstalación -> (False, ...) sin tocar nada
    - si no, descarga lo que falte, arranca, y sondea hasta timeout_arranque
    """
    if is_reachable():
        return True, "QuestDB ya estaba disponible"
    if not _es_host_local():
        return False, (f"QuestDB no responde en {QUESTDB_HOST}:{QUESTDB_PG_PORT} "
                       f"(host remoto configurado — revisa la conexión)")
    if platform.system() not in _SOS_SOPORTADOS:
        return False, ("Autoinstalación no soportada en este sistema operativo — "
                       "instala QuestDB manualmente")
    try:
        if not is_runtime_instalado():
            if progress_cb:
                progress_cb("Descargando QuestDB (una sola vez)…", None)
            descargar_runtime(progress_cb)
        if progress_cb:
            progress_cb("Arrancando QuestDB…", None)
        arrancar_bundled()
    except Exception as e:
        return False, f"No se pudo preparar QuestDB: {e}"

    inicio = time.time()
    while time.time() - inicio < timeout_arranque:
        if is_reachable():
            return True, "QuestDB arrancada"
        time.sleep(0.5)

    mensaje = "QuestDB no respondió a tiempo tras arrancarla"
    # en Mac, questdb.sh ya devolvió éxito con solo lanzar el proceso — si
    # luego el servidor se cae durante el boot, el motivo real solo queda en
    # su propio log, no en la salida de questdb.sh
    if platform.system() == 'Darwin':
        tail = _ultimo_log_questdb(_carpeta_datos())
        if tail:
            mensaje += f"\n\nÚltimas líneas del log de QuestDB:\n{tail}"
    return False, mensaje
