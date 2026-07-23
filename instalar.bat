@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo   Analytics System for Assets  -  Instalador
echo ==============================================
echo.
echo Este instalador se puede ejecutar tantas veces
echo como haga falta: solo instala lo que falte.
echo.

rem ---------- 1. Localizar Python 3.10+ ----------
set "PYCMD="
py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PYCMD=py -3"
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo No se ha encontrado Python 3.10 o superior.
    echo Intentando instalarlo automaticamente con winget...
    echo.
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo No se pudo instalar Python automaticamente.
        echo Descargalo desde https://www.python.org/downloads/
        echo ^(marcando la casilla "Add python.exe to PATH"^)
        echo y vuelve a ejecutar este instalador.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Python instalado. CIERRA esta ventana y vuelve a ejecutar
    echo instalar.bat ^(hace falta reabrir para que Windows encuentre Python^).
    echo.
    pause
    exit /b 0
)
for /f "delims=" %%v in ('%PYCMD% --version') do echo Python encontrado: %%v
echo.

rem ---------- 2. Crear el entorno virtual si no existe ----------
if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    %PYCMD% -m venv venv
    if errorlevel 1 (
        echo Error creando el entorno virtual.
        pause
        exit /b 1
    )
)

rem ---------- 3. Instalar/actualizar dependencias ----------
echo Instalando dependencias ^(la primera vez puede tardar varios minutos^)...
echo.
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Error instalando dependencias. Comprueba tu conexion a internet
    echo y vuelve a ejecutar este instalador.
    pause
    exit /b 1
)
copy /y requirements.txt "venv\.deps_instaladas" >nul

rem ---------- 4. Acceso directo en el escritorio ----------
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Analytics System.lnk'); $lnk.TargetPath = '%~dp0launcher.vbs'; $lnk.WorkingDirectory = '%~dp0'; $lnk.IconLocation = '%~dp0icon.ico'; $lnk.Save()"
if errorlevel 1 (
    echo Aviso: no se pudo crear el acceso directo del escritorio.
    echo Puedes abrir la app con doble clic en launcher.vbs
) else (
    echo Acceso directo "Analytics System" creado en el escritorio.
)

echo.
echo ==============================================
echo   Instalacion completada.
echo   Abre la app con el acceso directo
echo   "Analytics System" del escritorio
echo   ^(o con doble clic en launcher.vbs^).
echo ==============================================
echo.
pause
