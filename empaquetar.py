#!/usr/bin/env python3
"""
Empaquetador multiplataforma de Analytics System for Assets.

Uso:
    python empaquetar.py

Resultado en dist/:
    Windows  -> dist/AnalyticsSystem_Portable.zip  (Python embebido + app)
    macOS    -> dist/AnalyticsSystem.app           (.app autocontenido)
    Linux    -> dist/AnalyticsSystem_Portable.tar.gz  (AppImage simplificado)

El usuario final solo descomprime (Windows/Linux) o arrastra al Finder (Mac).
No necesita instalar Python ni dependencias.
"""

import os, sys, shutil, platform, subprocess, zipfile, tarfile, tempfile, urllib.request

PROJECT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(PROJECT, "dist")
CACHE = os.path.join(PROJECT, "cache")
APP_NAME = "AnalyticsSystem"

os.makedirs(DIST, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def shell(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def download(url, dest):
    if os.path.exists(dest):
        print(f"  [cache] {os.path.basename(dest)}")
        return
    print(f"  Descargando {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)


# ══════════════════════════════════════════════════════════════════
#  WINDOWS — PyInstaller con .exe único
# ══════════════════════════════════════════════════════════════════
def build_windows():
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Instalando PyInstaller...")
        shell(f'"{sys.executable}" -m pip install pyinstaller')

    app_py = os.path.join(PROJECT, "app.py")
    icon = os.path.join(PROJECT, "icon.ico")
    icon_arg = f' --icon="{icon}"' if os.path.exists(icon) else ""
    config_json = os.path.join(PROJECT, "config.json")
    config_arg = f' --add-data "{config_json};."' if os.path.exists(config_json) else ""

    workdir = os.path.join(DIST, "build_win")

    cmd = (
        f'"{sys.executable}" -m PyInstaller '
        f'--onefile --windowed --name="{APP_NAME}"'
        f'{icon_arg} '
        f'--add-data "{os.path.join(PROJECT, "core")};core" '
        f'--add-data "{os.path.join(PROJECT, "gui")};gui" '
        f'--add-data "{os.path.join(PROJECT, "library")};library"'
        f'{config_arg} '
        f'--collect-data matplotlib '
        f'--hidden-import=PyQt6.QtWebEngineWidgets '
        f'--hidden-import=PyQt6.QtWebEngineCore '
        f'--hidden-import=numba '
        f'--distpath="{DIST}" '
        f'--workpath="{workdir}" '
        f'--specpath="{workdir}" '
        f'"{app_py}"'
    )
    shell(cmd, cwd=PROJECT)

    exe = os.path.join(DIST, f"{APP_NAME}.exe")
    print(f"\n  >>> .exe creado: {exe}")
    print(f"  >>> Pesa aproximadamente {os.path.getsize(exe) / 1024 / 1024:.0f} MB")
    return exe


# ══════════════════════════════════════════════════════════════════
#  macOS — PyInstaller .app bundle
# ══════════════════════════════════════════════════════════════════
def build_macos():
    # Asegurar que PyInstaller está disponible
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Instalando PyInstaller...")
        shell(f'"{sys.executable}" -m pip install pyinstaller')

    # Crear .spec o usar línea de comandos
    app_py = os.path.join(PROJECT, "app.py")
    icon = os.path.join(PROJECT, "icon.ico")
    icon_arg = f' --icon="{icon}"' if os.path.exists(icon) else ""
    config_json = os.path.join(PROJECT, "config.json")
    config_arg = f' --add-data "{config_json}:."' if os.path.exists(config_json) else ""

    cmd = (
        f'"{sys.executable}" -m PyInstaller '
        f'--windowed --name="{APP_NAME}"'
        f'{icon_arg} '
        f'--add-data "{os.path.join(PROJECT, "core")}:core" '
        f'--add-data "{os.path.join(PROJECT, "gui")}:gui" '
        f'--add-data "{os.path.join(PROJECT, "library")}:library"'
        f'{config_arg} '
        f'--collect-data matplotlib '
        f'--hidden-import=PyQt6.QtWebEngineWidgets '
        f'--hidden-import=PyQt6.QtWebEngineCore '
        f'--hidden-import=numba '
        f'--distpath="{DIST}" '
        f'--workpath="{os.path.join(DIST, "build_mac")}" '
        f'--specpath="{os.path.join(DIST, "build_mac")}" '
        f'"{app_py}"'
    )
    shell(cmd, cwd=PROJECT)

    # Copiar carpetas vacías necesarias
    app_bundle = os.path.join(DIST, f"{APP_NAME}.app", "Contents", "MacOS")
    for d in ("Sistemas", "Sistemas/externos", "Favoritos"):
        os.makedirs(os.path.join(app_bundle, d), exist_ok=True)

    # Crear ZIP para distribución
    app_dir = os.path.join(DIST, f"{APP_NAME}.app")
    zip_dest = os.path.join(DIST, "AnalyticsSystem_macOS.zip")
    if os.path.exists(zip_dest):
        os.remove(zip_dest)
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(app_dir):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, DIST)
                zf.write(full, arcname)

    print(f"\n  >>> .app creado en: {app_dir}")
    print(f"  >>> ZIP creado:  {zip_dest}")
    return zip_dest


# ══════════════════════════════════════════════════════════════════
#  Linux — PyInstaller (tar.gz)
# ══════════════════════════════════════════════════════════════════
def build_linux():
    # Igual que macOS pero salida en carpeta plana
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Instalando PyInstaller...")
        shell(f'"{sys.executable}" -m pip install pyinstaller')

    app_py = os.path.join(PROJECT, "app.py")
    config_json = os.path.join(PROJECT, "config.json")
    config_arg = f' --add-data "{config_json}:."' if os.path.exists(config_json) else ""

    cmd = (
        f'"{sys.executable}" -m PyInstaller '
        f'--windowed --name="{APP_NAME}" '
        f'--add-data "{os.path.join(PROJECT, "core")}:core" '
        f'--add-data "{os.path.join(PROJECT, "gui")}:gui" '
        f'--add-data "{os.path.join(PROJECT, "library")}:library"'
        f'{config_arg} '
        f'--collect-data matplotlib '
        f'--hidden-import=PyQt6.QtWebEngineWidgets '
        f'--hidden-import=PyQt6.QtWebEngineCore '
        f'--hidden-import=numba '
        f'--distpath="{DIST}" '
        f'--workpath="{os.path.join(DIST, "build_linux")}" '
        f'--specpath="{os.path.join(DIST, "build_linux")}" '
        f'"{app_py}"'
    )
    shell(cmd, cwd=PROJECT)

    outdir = os.path.join(DIST, APP_NAME)
    for d in ("Sistemas", "Sistemas/externos", "Favoritos"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)

    tar_dest = os.path.join(DIST, "AnalyticsSystem_Linux.tar.gz")
    with tarfile.open(tar_dest, "w:gz") as tar:
        tar.add(outdir, arcname=os.path.basename(outdir))

    print(f"\n  >>> tar.gz creado: {tar_dest}")
    return tar_dest


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    system = platform.system()
    print(f"  SO detectado: {system}")
    print(f"  Destino: {DIST}")
    print()

    if system == "Windows":
        build_windows()
    elif system == "Darwin":
        build_macos()
    elif system == "Linux":
        build_linux()
    else:
        print(f"  [!] SO no soportado: {system}")
        sys.exit(1)

    print("\n  Empaquetado completado.")
