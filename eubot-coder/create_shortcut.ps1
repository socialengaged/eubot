$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:USERPROFILE\Desktop\Eubot Chat.lnk")
$s.TargetPath = "pythonw"
$s.Arguments = "local_server.py"
$s.WorkingDirectory = "C:\Users\info\progetti\eubot\eubot-coder"
$s.IconLocation = "C:\Users\info\progetti\eubot\eubot-coder\webapp\icon.ico,0"
$s.Description = "Eubot Coder Chat"
$s.Save()
Write-Host "Shortcut created on Desktop"
