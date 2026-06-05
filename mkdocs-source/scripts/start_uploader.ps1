$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

$Candidates = @()
if ($env:PYTHON) {
  $Candidates += $env:PYTHON
}
$Candidates += @("python", "py", $BundledPython)

$Python = $null
foreach ($Candidate in $Candidates) {
  try {
    $Version = & $Candidate --version 2>$null
    if ($LASTEXITCODE -eq 0 -and $Version) {
      $Python = $Candidate
      break
    }
  } catch {
    continue
  }
}

if (-not $Python) {
  throw "No usable Python executable was found. Install Python or set the PYTHON environment variable."
}

Push-Location $Root
try {
  Write-Host "Using Python: $Python"
  & $Python -m mkdocs --version *> $null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "MkDocs is not installed for this Python. Installing requirements.txt..."
    & $Python -m pip install -r requirements.txt
  }
  Write-Host "Starting ORL uploader at http://127.0.0.1:8765/"
  & $Python scripts\upload_server.py
} finally {
  Pop-Location
}
