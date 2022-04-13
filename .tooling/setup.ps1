copier -f -r 753f174
Remove-Item .venv -Recurse -ErrorAction SilentlyContinue
py -3.10 -m venv .venv --upgrade-deps
. .tooling/update.ps1
