$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$Pythonw = Join-Path $Root ".venv-build\Scripts\pythonw.exe"
$Executable = Join-Path $Root "Lyra.exe"
$IconPath = Join-Path $Root "assets\lyra_icon.ico"

if (Test-Path -LiteralPath $Pythonw -PathType Leaf) {
    $Target = $Pythonw
    $Arguments = "-m lyra"
} elseif (Test-Path -LiteralPath $Executable -PathType Leaf) {
    $Target = $Executable
    $Arguments = ""
} else {
    throw "Lyra runtime was not found. Create .venv-build or build Lyra.exe first."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Lyra.lnk"
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Target
$Shortcut.Arguments = $Arguments
$Shortcut.WorkingDirectory = $Root
$Shortcut.IconLocation = "$IconPath,0"
$Shortcut.Description = "Lyra electronic-structure plotting studio"
$Shortcut.Save()

Write-Output $ShortcutPath
