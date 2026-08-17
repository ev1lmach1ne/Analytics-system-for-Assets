# consolidar_noche.ps1
# Job nocturno LOCAL: sincroniza el repo y consolida en Qdrant el resumen
# diario del día anterior (etiqueta 'resumen_diario').
#
# Complementa el cron de GitHub Actions (consolidacion_diaria.yml), que a las
# 00:00 UTC genera y commitea el JSON del día en memoria/resumenes_diarios/.
# Este script (que sí tiene acceso a la BD vectorial local) tira del repo,
# toma los resúmenes de cambio del día y vuelca el punto consolidado en Qdrant.
#
# Registrado como tarea programada de Windows:
#   schtasks /Create /TN "AnalyticsSystem-MemoriaDiaria" /TR "powershell -ExecutionPolicy Bypass -File \"...\consolidar_noche.ps1\"" /SC DAILY /ST 02:00 /F
# Para desinstalarlo:
#   schtasks /Delete /TN "AnalyticsSystem-MemoriaDiaria" /F

$ErrorActionPreference = "Stop"

$PROJECT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT

Write-Output "[$(Get-Date -Format o)] Sincronizando repo..."
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) {
    Write-Output "[$(Get-Date -Format o)] git pull falló; se intenta igualmente consolidar con lo local."
}

Write-Output "[$(Get-Date -Format o)] Consolidando resumen diario (día anterior) e indexando en Qdrant..."
& "$PROJECT\venv\Scripts\python.exe" "$PROJECT\memoria\consolidar_diario.py" --indexar
$exit = $LASTEXITCODE
Write-Output "[$(Get-Date -Format o)] Fin. Exit code: $exit"
exit $exit
