$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python311 = 'C:\Users\52834\AppData\Local\Programs\Python\Python311\python.exe'
$venvPath = Join-Path $root '.venv_sionna'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$requirements = Join-Path $root 'requirements-sionna-py311.txt'

if (-not (Test-Path $python311)) {
    throw "Python 3.11 not found at: $python311"
}

if (-not (Test-Path $venvPython)) {
    & $python311 -m venv $venvPath
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $requirements

Write-Host "Sionna environment ready:" $venvPython
