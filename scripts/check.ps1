$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SrcPath = Join-Path $RepoRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$env:PYTHONPATH;$SrcPath"
} else {
    $env:PYTHONPATH = $SrcPath
}

Set-Location $RepoRoot

python -m utils.ensure_nltk_data

python -m compileall -q src
python -m pytest
