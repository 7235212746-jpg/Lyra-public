param(
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$Venv = Join-Path $Root ".venv-build"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    $PythonCommand = Get-Command -Name $PythonPath -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python 3.12 or newer was not found. Supply -PythonPath with your Python executable."
    }
    $PythonPath = $PythonCommand.Source
}

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    & $PythonPath -m venv $Venv
}

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e "$Root[full]" pyinstaller

Push-Location $Root
try {
    & $VenvPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name Lyra `
        --icon "assets\lyra_icon.ico" `
        --add-data "assets\lyra_icon.png;assets" `
        --add-data "lyra\resources;lyra\resources" `
        --add-data "lyra\remote\winscp_bridge.ps1;lyra\remote" `
        --hidden-import "matplotlib.backends.backend_pdf" `
        --hidden-import "matplotlib.backends.backend_svg" `
        --collect-all skimage `
        --collect-all pymatgen `
        "lyra\__main__.py"
    Copy-Item -LiteralPath (Join-Path $Root "dist\Lyra.exe") -Destination (Join-Path $Root "Lyra.exe") -Force
}
finally {
    Pop-Location
}

Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Root "Lyra.exe")
