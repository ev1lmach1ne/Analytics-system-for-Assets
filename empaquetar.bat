@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Empaquetar Analytics System

echo ==============================================
echo   Empaquetar - Analytics System for Assets
echo   Crea una version portable sin instalacion
echo ==============================================
echo.

rem ── 1. Encontrar Python (para construir el paquete, no para el resultado) ──
set "PYCMD="
py -3 -c "print(1)" >nul 2>&1 && set "PYCMD=py -3"
if not defined PYCMD python -c "print(1)" >nul 2>&1 && set "PYCMD=python"
if not defined PYCMD (
    echo [!] No se encontro Python. Necesitas Python 3.10+ para ejecutar este script.
    echo     Instalalo desde https://www.python.org/downloads/
    pause & exit /b 1
)
for /f "delims=" %%v in ('%PYCMD% -c "import sys; print(sys.version)"') do echo [*] Python: %%v

rem ── 2. Version de Python portable (siempre 3.12 por estabilidad) ──
set "PYVER=3.12"
set "PYVERNODOT=312"
set "PYFULLVER=3.12.9"
set "PYARCH=amd64"
echo [*] Python portable: %PYFULLVER% %PYARCH%

rem ── 3. Directorios ──
set "DIST=dist"
set "APPDIR=%DIST%\AnalyticsSystem"
set "PYDIR=%APPDIR%\python"
set "CACHE=cache"

if not exist "%CACHE%" mkdir "%CACHE%"

rem ── 4. Descargar Python embeddable (si no esta en cache) ──
set "PYZIP=python-%PYFULLVER%-embed-%PYARCH%.zip"
set "PYURL=https://www.python.org/ftp/python/%PYFULLVER%/%PYZIP%"

if not exist "%CACHE%\%PYZIP%" (
    echo [*] Descargando Python portable...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PYURL%' -OutFile '%CACHE%\%PYZIP%' -UseBasicParsing"
    if errorlevel 1 (
        echo [!] Error descargando Python. Comprueba tu conexion a internet.
        echo     URL: %PYURL%
        pause & exit /b 1
    )
)

rem ── 5. Extraer Python embeddable ──
if exist "%PYDIR%" (
    echo [*] Limpiando python portable anterior...
    rmdir /s /q "%PYDIR%" 2>nul
)
echo [*] Extrayendo Python portable...
mkdir "%PYDIR%" 2>nul
powershell -NoProfile -Command "Expand-Archive -Path '%CACHE%\%PYZIP%' -DestinationPath '%PYDIR%' -Force"
if errorlevel 1 (
    echo [!] Error extrayendo Python portable.
    pause & exit /b 1
)

rem ── 6. Configurar Python embeddable para pip ──
echo [*] Configurando Python portable...
set "PTH=%PYDIR%\python%PYVERNODOT%._pth"
if exist "%PTH%" (
    rem Descomentar import site para habilitar pip
    powershell -NoProfile -Command "(Get-Content '%PTH%') -replace '#import site', 'import site' | Set-Content '%PTH%'"
    rem Añadir Lib\site-packages
    echo Lib\site-packages>> "%PTH%"
)

rem ── 7. Instalar pip en Python portable ──
set "PIPBOOT=%CACHE%\get-pip.py"
if not exist "%PIPBOOT%" (
    echo [*] Descargando get-pip.py...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PIPBOOT%' -UseBasicParsing"
)
echo [*] Instalando pip...
"%PYDIR%\python.exe" "%PIPBOOT%" --no-warn-script-location >nul 2>&1
if errorlevel 1 (
    echo [!] Error instalando pip.
    pause & exit /b 1
)

rem ── 8. Instalar setuptools (necesario para paquetes desde src) ──
echo [*] Instalando setuptools y wheel...
"%PYDIR%\python.exe" -m pip install setuptools wheel --no-warn-script-location >nul 2>&1

rem ── 9. Instalar dependencias en Python portable ──
echo [*] Instalando dependencias (esto tarda unos minutos)...
"%PYDIR%\python.exe" -m pip install -r requirements.txt --no-warn-script-location --trusted-host pypi.org --trusted-host files.pythonhosted.org
if errorlevel 1 (
    echo [!] Error instalando dependencias.
    pause & exit /b 1
)

rem ── 10. Copiar archivos de la app ──
echo [*] Copiando archivos de la aplicacion...

rem Limpiar dist anterior (conservar python si ya esta)
if exist "%APPDIR%\app.py" del /q "%APPDIR%\app.py" 2>nul
if exist "%APPDIR%\core" rmdir /s /q "%APPDIR%\core" 2>nul
if exist "%APPDIR%\gui" rmdir /s /q "%APPDIR%\gui" 2>nul
if exist "%APPDIR%\library" rmdir /s /q "%APPDIR%\library" 2>nul
if exist "%APPDIR%\tests" rmdir /s /q "%APPDIR%\tests" 2>nul
if exist "%APPDIR%\Sistemas" rmdir /s /q "%APPDIR%\Sistemas" 2>nul
if exist "%APPDIR%\Favoritos" rmdir /s /q "%APPDIR%\Favoritos" 2>nul

rem Copiar codigo fuente
xcopy /q /e /i core "%APPDIR%\core" >nul
xcopy /q /e /i gui "%APPDIR%\gui" >nul
xcopy /q /e /i library "%APPDIR%\library" >nul
xcopy /q /e /i tests "%APPDIR%\tests" >nul
copy /y app.py "%APPDIR%\app.py" >nul
copy /y config.json "%APPDIR%\config.json" >nul 2>nul
copy /y icon.ico "%APPDIR%\icon.ico" >nul 2>nul

rem Copiar data (si existe la carpeta data con Limpiados etc.)
if exist "data" xcopy /q /e /i data "%APPDIR%\data" >nul 2>nul

rem Crear carpetas vacias necesarias
mkdir "%APPDIR%\Sistemas" 2>nul
mkdir "%APPDIR%\Sistemas\externos" 2>nul
mkdir "%APPDIR%\Favoritos" 2>nul

rem ── 11. Crear launcher ──
echo [*] Creando launcher...
(
echo @echo off
echo cd /d "%%~dp0"
echo start "" "%%~dp0python\pythonw.exe" "%%~dp0app.py"
) > "%APPDIR%\Iniciar.bat"

rem Launcher VBS (sin ventana de consola)
(
echo Set sh = CreateObject^("WScript.Shell"^)
echo sh.CurrentDirectory = CreateObject^("Scripting.FileSystemObject"^).GetParentFolderName^(WScript.ScriptFullName^)
echo sh.Run "python\pythonw.exe app.py", 0, False
) > "%APPDIR%\Iniciar.vbs"

rem ── 12. Crear ZIP ──
echo [*] Creando archivo comprimido...
set "ZIPFILE=%DIST%\AnalyticsSystem_Portable.zip"
if exist "%ZIPFILE%" del "%ZIPFILE%" 2>nul
powershell -NoProfile -Command "Compress-Archive -Path '%APPDIR%\*' -DestinationPath '%ZIPFILE%' -Force"
if errorlevel 1 (
    echo [!] Error creando el ZIP.
    echo     La carpeta portable esta en: %CD%\%APPDIR%
    pause & exit /b 1
)

rem ── 13. Resumen ──
echo.
echo ==============================================
echo   Empaquetado completado
echo.
echo   ZIP portable: dist\AnalyticsSystem_Portable.zip
echo.
echo   Para el usuario final:
echo     1. Descomprimir el ZIP donde quiera
echo     2. Doble clic en Iniciar.vbs
echo     3. Listo, no necesita instalar nada
echo.
echo   Para subir una actualizacion:
echo     1. Haz los cambios en el codigo
echo     2. Vuelve a ejecutar empaquetar.bat
echo     3. El ZIP se regenera solo
echo ==============================================
pause
