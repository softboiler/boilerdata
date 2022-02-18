copier -f -r a578b9f
Remove-Item .venv -Recurse -ErrorAction SilentlyContinue
py -3.10 -m venv .venv
. tooling/update.ps1
