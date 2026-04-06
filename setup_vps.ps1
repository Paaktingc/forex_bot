$ErrorActionPreference = "Stop"

$pythonInstallDir = "C:\Python311"
$repoUrl = "https://github.com/USERNAME/forex_bot"
$repoDir = "C:\forex_bot"
$pythonExe = Join-Path $pythonInstallDir "python.exe"
$pipExe = Join-Path $pythonInstallDir "Scripts\pip.exe"
$startupScript = Join-Path $repoDir "run_bot.bat"
$taskName = "ForexBotStartup"

Write-Host "Installing Python 3.11..."
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements

Write-Host "Installing Git..."
winget install Git.Git --silent --accept-package-agreements --accept-source-agreements

Write-Host "Waiting for Python install to settle..."
Start-Sleep -Seconds 5

if (-not (Test-Path $pythonExe)) {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path $pipExe)) {
    $pipExe = "$pythonExe -m pip"
}

if (-not (Test-Path $repoDir)) {
    Write-Host "Cloning repository..."
    git clone $repoUrl $repoDir
} else {
    Write-Host "Repository already exists at $repoDir. Skipping clone."
}

Set-Location $repoDir

Write-Host "Installing Python requirements..."
if (Test-Path $pipExe) {
    & $pipExe install -r requirements.txt
} else {
    & $pythonExe -m pip install -r requirements.txt
}

Write-Host "Creating .env placeholder file..."
@"
MT5_LOGIN=REPLACE_WITH_YOUR_LOGIN
MT5_PASSWORD=REPLACE_WITH_YOUR_PASSWORD
MT5_SERVER=REPLACE_WITH_SERVER_NAME
TELEGRAM_BOT_TOKEN=REPLACE_WITH_TOKEN
TELEGRAM_CHAT_ID=REPLACE_WITH_CHAT_ID
"@ | Out-File -FilePath (Join-Path $repoDir ".env") -Encoding UTF8

Write-Host "Creating startup launcher..."
@"
@echo off
cd /d C:\forex_bot
"$pythonExe" C:\forex_bot\main.py
"@ | Out-File -FilePath $startupScript -Encoding ASCII

Write-Host "Registering Windows Task Scheduler task..."
$action = New-ScheduledTaskAction -Execute $startupScript
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next steps:"
Write-Host "1. Edit C:\forex_bot\.env with real MT5 and Telegram credentials."
Write-Host "2. Replace USERNAME in setup_vps.ps1 with your GitHub username or repo URL."
Write-Host "3. Open MetaTrader 5 on the VPS and confirm your trading account is logged in."
Write-Host "4. Run: $pythonExe C:\forex_bot\health_check.py"
Write-Host "5. Reboot once or start the '$taskName' task manually to verify auto-start."
