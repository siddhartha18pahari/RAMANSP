# Build main.pdf.
#
#   powershell -ExecutionPolicy Bypass -File paper\build.ps1
#
# Finds a TeX installation (TinyTeX, MiKTeX or anything already on PATH),
# installs any missing LaTeX packages when the distribution supports it, and
# runs the pdflatex/bibtex/pdflatex/pdflatex sequence that resolves both
# citations and cross-references.

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
    Write-Output "No TeX installation found. Install one of:"
    Write-Output "  TinyTeX : https://yihui.org/tinytex/  (no admin required)"
    Write-Output "  MiKTeX  : winget install --id MiKTeX.MiKTeX"
    Write-Output "Or upload main.tex, refs.bib, figures/ and tables/ to Overleaf."
    exit 1
}
Write-Output "using TeX at $bin"
$env:PATH = "$bin;$env:PATH"

# TinyTeX and TeX Live ship a minimal package set; add what this document needs.
$tlmgr = Join-Path $bin "tlmgr.bat"
if (Test-Path $tlmgr) {
    $pkgs = @("algorithms", "algorithmicx", "booktabs", "authblk", "siunitx",
              "lm", "geometry", "hyperref", "xcolor", "amsmath", "graphics",
              "caption", "etoolbox", "l3packages", "l3kernel", "translator",
              "float", "psnfss", "carlisle", "ec")
    Write-Output "ensuring LaTeX packages: $($pkgs -join ', ')"
    & $tlmgr install @pkgs 2>&1 | Select-String -Pattern "install|already|error" |
        Select-Object -First 12 | ForEach-Object { Write-Output ("  " + $_) }
}

Write-Output "pass 1 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode -halt-on-error main.tex > build1.log 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output "pdflatex failed; last errors:"
    Select-String -Path build1.log -Pattern "^!|^l\.\d" | Select-Object -First 20 |
        ForEach-Object { Write-Output ("  " + $_.Line) }
    exit 1
}
Write-Output "pass 2 of 4 (bibtex)"
& bibtex main > build2.log 2>&1
Write-Output "pass 3 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode main.tex > build3.log 2>&1
Write-Output "pass 4 of 4 (pdflatex)"
& pdflatex -interaction=nonstopmode main.tex > build4.log 2>&1

if (Test-Path "main.pdf") {
    $kb = [math]::Round((Get-Item "main.pdf").Length / 1KB, 1)
    $pages = (Select-String -Path build4.log -Pattern "Output written .*\((\d+) pages" |
              ForEach-Object { $_.Matches[0].Groups[1].Value }) | Select-Object -Last 1
    Write-Output "BUILT main.pdf  ($kb KB, $pages pages)"
    $warn = Select-String -Path build4.log -Pattern "Warning: Citation|Warning: Reference" |
            Select-Object -First 8
    if ($warn) {
        Write-Output "unresolved references:"
        $warn | ForEach-Object { Write-Output ("  " + $_.Line.Trim()) }
    } else {
        Write-Output "all citations and cross-references resolved"
    }
} else {
    Write-Output "no main.pdf produced; see build*.log"
    exit 1
}
