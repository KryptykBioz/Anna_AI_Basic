# start.ps1

# Change to script directory
Set-Location -Path $PSScriptRoot

# Activate the virtual environment
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Run the GUI frontend instead of the terminal bot
# python ./BASE/interface/gui.py




# ALT: Paste the following into console to run venv
# ./venv/Scripts/Activate.ps1

# Navigate to the project root directory (Toma_AI)
Set-Location -Path $PSScriptRoot

# Start the Python GUI application using its relative path from the project root
python .\BASE\interface\gui.py

# When gui.py exits, keep the window open
Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyUp")