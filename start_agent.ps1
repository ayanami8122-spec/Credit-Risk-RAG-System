param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path $Root "agent"

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Host "Root .env was not found. Copy .env.example to .env and fill your API key first." -ForegroundColor Yellow
}

if (-not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    Write-Host "streamlit was not found. Run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

Push-Location $AppDir
try {
    streamlit run app.py --server.port $Port
}
finally {
    Pop-Location
}
