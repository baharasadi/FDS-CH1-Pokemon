$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"
$Python = Get-Command py -ErrorAction SilentlyContinue

if ($Python) {
    & py -3 -m venv $VenvDir
} else {
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $Python) {
        throw "Python 3 is required. Install Python 3 and try again."
    }
    & python -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $RootDir "requirements.txt")

Write-Host "Environment created at $VenvDir"
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
