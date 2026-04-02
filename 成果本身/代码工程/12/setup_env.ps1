$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python38 = 'C:\Users\52834\AppData\Local\Programs\Python\Python38\python.exe'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$requirements = Join-Path $root 'requirements-pipeline-py38.txt'

if (-not (Test-Path $python38)) {
    throw "Python 3.8 not found at: $python38"
}

if (-not (Test-Path (Join-Path $root '.venv\Scripts\python.exe'))) {
    & $python38 -m venv (Join-Path $root '.venv')
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $requirements

Write-Host "Environment ready:" $venvPython
