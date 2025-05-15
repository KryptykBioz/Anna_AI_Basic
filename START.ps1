# start.ps1

# Change to script directory
Set-Location -Path $PSScriptRoot

# Activate the virtual environment
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# .\venv\Scripts\Activate.ps1

# Run the bot script
python "$PSScriptRoot\main.py"

# When main.py exits, keep the window open
Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")