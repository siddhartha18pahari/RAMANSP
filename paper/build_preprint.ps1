# Build preprint.pdf, the ordinary-format version of the same paper.
#
#   powershell -ExecutionPolicy Bypass -File paper\build_preprint.ps1
#
# Same TeX installation and the same four-pass sequence as build.ps1, and the
# same temporary job name so an open viewer cannot kill the run.

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

$tlmgr = Join-Path $bin "tlmgr.bat"
if (Test-Path $tlmgr) {
    & $tlmgr install natbib abstract 2>&1 |
        Select-String -Pattern "install|already|error" |
        Select-Object -First 4 | ForEach-Object { Write-Output ("  " + $_) }
}

Write-Output "pass 1 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode -halt-on-error -jobname pp_tmp preprint.tex > pp_build1.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output "pdflatex failed; last errors:"
    Select-String -Path pp_build1.log -Pattern "^!|^l\.\d" | Select-Object -First 20 |
        ForEach-Object { Write-Output ("  " + $_.Line) }
    exit 1
}
Write-Output "pass 2 of 4 (bibtex)"
& bibtex pp_tmp > pp_build2.log 2>&1
Write-Output "pass 3 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode -jobname pp_tmp preprint.tex > pp_build3.log 2>&1
Write-Output "pass 4 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode -jobname pp_tmp preprint.tex > pp_build4.log 2>&1

$built = $null
if (Test-Path "pp_tmp.pdf") {
    try {
        Move-Item -Force "pp_tmp.pdf" "preprint.pdf" -ErrorAction Stop
        $built = "preprint.pdf"
    } catch {
        Write-Output "preprint.pdf is open elsewhere; the fresh build is pp_tmp.pdf"
        $built = "pp_tmp.pdf"
    }
}

if ($built) {
    $kb = [math]::Round((Get-Item $built).Length / 1KB, 1)
    $pages = (Select-String -Path pp_build4.log -Pattern "Output written .*\((\d+) pages" |
              ForEach-Object { $_.Matches[0].Groups[1].Value }) | Select-Object -Last 1
    Write-Output "BUILT $built  ($kb KB, $pages pages)"
    $warn = Select-String -Path pp_build4.log -Pattern "Warning: Citation|Warning: Reference" |
            Select-Object -First 8
    if ($warn) {
        Write-Output "unresolved references:"
        $warn | ForEach-Object { Write-Output ("  " + $_.Line.Trim()) }
    } else {
        Write-Output "all citations and cross-references resolved"
    }
} else {
    Write-Output "no preprint.pdf produced; see pp_build*.log"
    exit 1
}
