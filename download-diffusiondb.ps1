# Download numbered DiffusionDB archives with the project's own downloader.
# Example: powershell -ExecutionPolicy Bypass -File .\download-diffusiondb.ps1 -Start 1 -End 3 -Unzip
[CmdletBinding()]
param(
    [ValidateRange(1, 14000)] [int] $Start = 1,
    [ValidateRange(1, 14000)] [int] $End = 1,
    [switch] $Large,
    [switch] $Unzip,
    [string] $OutputDirectory,
    [string] $SourceDirectory,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$maxPart = if ($Large) { 14000 } else { 2000 }
if ($Start -gt $End -or $End -gt $maxPart) {
    throw "Choose an inclusive range from 1 to $maxPart with Start <= End."
}
if (-not $OutputDirectory) {
    $OutputDirectory = if ($Large) { 'D:\DiffusionDB\large' } else { 'D:\DiffusionDB\images' }
}
if (-not $SourceDirectory) {
    $SourceDirectory = Join-Path $PSScriptRoot 'diffusiondb-source'
}

$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$SourceDirectory = [IO.Path]::GetFullPath($SourceDirectory)
if ([IO.Path]::GetPathRoot($OutputDirectory) -ine 'D:\') {
    throw "OutputDirectory must be on drive D: $OutputDirectory"
}

$downloadScript = Join-Path $SourceDirectory 'scripts\download.py'
$venvPython = Join-Path $SourceDirectory '.venv\Scripts\python.exe'
$partPrefix = if ($Large) { 'DiffusionDB Large' } else { 'DiffusionDB 2M' }
Write-Host "$partPrefix parts $Start through $End (inclusive) -> $OutputDirectory"

if ($DryRun) {
    if (-not (Test-Path -LiteralPath $downloadScript)) {
        Write-Host "Would clone https://github.com/poloclub/diffusiondb.git into $SourceDirectory"
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Host "Would create a Python virtual environment and install alive-progress in $SourceDirectory\.venv"
    }
    $preview = "python scripts/download.py -i $Start -r $($End + 1) -o `"$OutputDirectory`""
    if ($Large) { $preview += ' -l' }
    Write-Host "Would run: $preview"
    if ($Unzip) { Write-Host "Would extract the downloaded ZIP files into $OutputDirectory" }
    return
}

if (-not (Test-Path -LiteralPath $downloadScript)) {
    if (Test-Path -LiteralPath $SourceDirectory) {
        throw "SourceDirectory exists but scripts/download.py is missing: $SourceDirectory"
    }
    git clone --depth 1 https://github.com/poloclub/diffusiondb.git $SourceDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone the DiffusionDB repository.' }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    py -3 -m venv (Join-Path $SourceDirectory '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    & $venvPython -m pip install alive-progress
    if ($LASTEXITCODE -ne 0) { throw 'Could not install alive-progress.' }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$arguments = @($downloadScript, '-i', $Start, '-r', ($End + 1), '-o', $OutputDirectory)
if ($Large) { $arguments += '-l' }

# The upstream range end is exclusive. Its single-file -z path and extraction
# directory handling are unreliable, so always use range mode and unzip here.
Push-Location $OutputDirectory
try {
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "DiffusionDB downloader exited with code $LASTEXITCODE." }

    for ($part = $Start; $part -le $End; $part++) {
        $zipPath = Join-Path $OutputDirectory ('part-{0:D6}.zip' -f $part)
        if (-not (Test-Path -LiteralPath $zipPath) -or (Get-Item -LiteralPath $zipPath).Length -eq 0) {
            throw "Missing or empty archive: $zipPath"
        }
        if ($Unzip) {
            Write-Host "Extracting $zipPath"
            Expand-Archive -LiteralPath $zipPath -DestinationPath $OutputDirectory -Force
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Done. Archives and any extracted images are in $OutputDirectory"
