copier -r 9c1800d
py -3.10 -m venv .venv
.venv/Scripts/activate
pip install -U pip  # throws [WinError 5], but still works on its own
pip install wheel
pip install -r tooling/requirements_dev.txt
flit install -s
