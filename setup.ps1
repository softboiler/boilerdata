copier -f -r b2770eb
Remove-Item .venv -Recurse -ErrorAction SilentlyContinue
py -3.10 -m venv .venv --upgrade-deps
. .tools/setup/update.ps1
