# Python is assumed to be available in the system PATH
$pythonPath = "python"

# List of Python scripts to execute sequentially
$scripts = @(
    "sieve_eratosthenes.py",
    "sieve_eratosthenes_cuda.py",
    "sieve_eratosthenes_multiprocessing.py",
    "trial_division.py",
    "trial_division_cuda.py",
    "trial_division_multiprocessing.py"
)

# Execute each script in sequence
foreach ($script in $scripts) {
    # Print a header for better separation in the output
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " Starting Script: $script " -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    # Run the Python script
    & $pythonPath $script

    # Check the exit code of the script
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Script $script encountered an issue. Exiting the process." -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        break
    }

    # Print success message
    Write-Host ""
    Write-Host "Script $script completed successfully." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}
