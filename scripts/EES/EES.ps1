# Output will be "NA" if the EES script failed for some reason
Set-Clipboard "NA"

# Open the specified file in EES, waiting for it to finish opening
& $env:EES $env:EESFILE /nosplash
Start-Sleep 3

# Focus the window, send keypresses to solve the file and get output
$wshell = New-Object -ComObject wscript.shell

# Uncomment one of these depending on whether you are at home or at work
$wshell.AppActivate("EES Professional:  ${Env:EESFILE}") | Out-Null
# $wshell.AppActivate("EES Commercial Version:  ${Env:EESFILE}") | Out-Null
# $wshell.AppActivate("EES Academic Commercial:  ${Env:EESFILE}") | Out-Null

.\Send-Key.ps1 -KeyCode @(0x71)  # Send "F2" to solve the EES file
Start-Sleep 0.1  # Wait long enough for EES to solve
.\Send-Key.ps1 -KeyCode @(0x11, 0x51)  # Send "CTRL+Q" to quit EES

# The last line of the EES file wrote results to the clipboard
Get-Clipboard | Out-File -FilePath .\out.dat
