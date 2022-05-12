copier -f -r 7a46ec1
Remove-Item .venv -Recurse -ErrorAction SilentlyContinue
py -3.10 -m venv .venv --upgrade-deps
. update.ps1
