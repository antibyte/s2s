# Stop local Qwen tts-server (frees VRAM/driver load when not needed).
Get-Process tts-server -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping tts-server pid=$($_.Id)"
    Stop-Process -Id $_.Id -Force
}
# Also match if renamed / full path
Get-CimInstance Win32_Process -Filter "Name='tts-server.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "Done. GPU/VRAM from Qwen should release shortly."
