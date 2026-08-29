# Regenerate every downstream artefact and build the PDF.
#
#   powershell -ExecutionPolicy Bypass -File run\build_all.ps1
#
# Assumes 01_build_corpus, 03_splat_fit and 07_ml_benchmark have already run;
# those are the expensive stages. This refreshes the graph, the figures, the
# LaTeX macros and tables, checks anonymity, and compiles the manuscript.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Step($label, $cmd) {
    Write-Output ""
    Write-Output ("=== " + $label + " ===")
    & python -u $cmd
    if ($LASTEXITCODE -ne 0) { throw "$label failed" }
}

Step "02 knowledge graph"  "run/02_knowledge_graph.py"
Step "04 figures"          "run/04_figures.py"
Step "06 workflow figure"  "run/06_workflow_figure.py"
Step "05 paper assets"     "run/05_paper_assets.py"

Write-Output ""
Write-Output "=== anonymity gate ==="
& python -u "run/check_anonymity.py"
if ($LASTEXITCODE -ne 0) { throw "anonymity gate failed: refusing to build" }

Write-Output ""
Write-Output "=== manuscript ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File "paper\build.ps1"
