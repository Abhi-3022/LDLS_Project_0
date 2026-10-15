# run_tests.ps1 — run the driver on a document and record CPU run time.
# Usage (from the llama.cpp folder, or edit $Driver/$Model below):
#   powershell -ExecutionPolicy Bypass -File ..\run_tests.ps1 -Doc ..\dog.json -Ask 3 -Change 6 -Evict 5
#
# It runs the two built-in tests (TEST 1 isolation, TEST 2 eviction) and prints the
# wall-clock time, so you have the CPU run-time number for the report slide.

param(
  [string]$Model  = "models\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  [string]$Driver = "build\bin\Release\llama-doc-driver.exe",
  [Parameter(Mandatory=$true)][string]$Doc,
  [int]$Ask    = 3,
  [int]$Change = 6,
  [int]$Evict  = 5
)

Write-Host "=== Running driver on $Doc  (ask=$Ask change=$Change evict=$Evict) ===" -ForegroundColor Cyan

$elapsed = Measure-Command {
  & $Driver $Model $Doc --ask $Ask --change $Change --evict $Evict 2>&1 | Tee-Object -Variable out
}

# echo the driver output, then the timing
$out | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host ("CPU run time: {0:N2} s" -f $elapsed.TotalSeconds) -ForegroundColor Green
