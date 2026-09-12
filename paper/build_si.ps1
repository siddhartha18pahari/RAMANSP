# Build si.pdf, the Supporting Information document.
#
#   powershell -ExecutionPolicy Bypass -File paper\build_si.ps1
#
# Uses the same TeX installation as build.ps1. The SI cites nothing, so two
# pdflatex passes are enough to settle its cross-references.

$ErrorActionPreference = "Continue"
$paper = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $paper

function Find-Tex {
    $candidates = @(
        "$env:APPDATA\TinyTeX\bin\windows",
        "$env:APPDATA\TinyTeX\bin\win32",
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
        "C:\Program Files\MiKTeX\miktex\bin\x64",
        "C:\texlive\2025\bin\windows",
        "C:\texlive\2024\bin\windows"
    )
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c "pdflatex.exe")) { return $c }
    }
    $onPath = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($onPath) { return (Split-Path -Parent $onPath.Source) }
    return $null
}

$bin = Find-Tex
if (-not $bin) {
    Write-Output "No TeX installation found; see paper\build.ps1 for options."
    exit 1
}
$env:PATH = "$bin;$env:PATH"
Write-Output "using TeX at $bin"

# Build under a temporary job name, then move into place. pdflatex cannot
# overwrite a PDF that a viewer holds open, and failing the whole build for
# that reason wastes a run; this way the fresh PDF always exists somewhere.
Write-Output "pass 1 of 2 (pdflatex)"
& pdflatex -interaction=nonstopmode -halt-on-error -jobname si_tmp si.tex > si_build1.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output "pdflatex failed; last errors:"
    Select-String -Path si_build1.log -Pattern "^!|^l\.\d" | Select-Object -First 20 |
        ForEach-Object { Write-Output ("  " + $_.Line) }
    exit 1
}
Write-Output "pass 2 of 2 (pdflatex)"
& pdflatex -interaction=nonstopmode -jobname si_tmp si.tex > si_build2.log 2>&1

$built = $null
if (Test-Path "si_tmp.pdf") {
    try {
        Move-Item -Force "si_tmp.pdf" "si.pdf" -ErrorAction Stop
        $built = "si.pdf"
    } catch {
        Write-Output "si.pdf is open in another program; the fresh build is si_tmp.pdf"
        $built = "si_tmp.pdf"
    }
} elseif (Test-Path "si.pdf") {
    $built = "si.pdf"
}
if ($built) {
    $kb = [math]::Round((Get-Item $built).Length / 1KB, 1)
    $pages = (Select-String -Path si_build2.log -Pattern "Output written .*\((\d+) pages" |
              ForEach-Object { $_.Matches[0].Groups[1].Value }) | Select-Object -Last 1
    Write-Output "BUILT $built  ($kb KB, $pages pages)"
    $warn = Select-String -Path si_build2.log -Pattern "Warning: Reference" |
            Select-Object -First 8
    if ($warn) {
        Write-Output "unresolved references:"
        $warn | ForEach-Object { Write-Output ("  " + $_.Line.Trim()) }
    } else {
        Write-Output "all cross-references resolved"
    }
} else {
    Write-Output "no si.pdf produced; see si_build*.log"
    exit 1
}
