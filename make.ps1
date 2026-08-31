# Windows stand-in for GNU make. Same target names as the Makefile, so the
# documented commands work on this machine without installing make.
#   .\make.ps1 test
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'test', 'lint', 'imports', 'hooks', 'eval', 'clean')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# uv may be on PATH or installed into the ambient Python; prefer the former.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uv = @('uv')
} else {
    $uv = @('python', '-m', 'uv')
}

function Invoke-Uv {
    param([string[]]$Arguments)
    # Note: $uv[1..($uv.Count - 1)] is WRONG when $uv has one element - PowerShell
    # reads 1..0 as a descending range and yields @($null, $uv[0]).
    $exe = $uv[0]
    $prefix = @()
    if ($uv.Count -gt 1) { $prefix = $uv[1..($uv.Count - 1)] }
    & $exe @($prefix + $Arguments)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    'help' {
        Write-Host 'setup   - create the uv environment from uv.lock'
        Write-Host 'test    - run the test suite'
        Write-Host 'lint    - run all pre-commit hooks over every file'
        Write-Host 'imports - check layer boundaries with import-linter'
        Write-Host 'hooks   - install the git pre-commit hooks'
        Write-Host 'eval    - produce reports/eval_<git-sha>.md (phase 2)'
    }
    'setup'   { Invoke-Uv @('sync', '--extra', 'dev') }
    'test'    { Invoke-Uv @('run', 'pytest') }
    'imports' { Invoke-Uv @('run', 'lint-imports') }
    'lint'    { Invoke-Uv @('run', 'pre-commit', 'run', '--all-files') }
    'hooks'   { Invoke-Uv @('run', 'pre-commit', 'install') }
    'eval' {
        Write-Host 'make eval lands in phase 2, together with B0 and B1.'
        Write-Host 'No model exists yet, and no results table exists yet. That is correct.'
        exit 1
    }
    'clean' {
        Get-ChildItem -Recurse -Directory -Include '__pycache__', '.pytest_cache' |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}
