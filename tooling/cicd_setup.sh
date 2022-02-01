pip install -U pip wheel
pip install pipx
pipx install copier==6.0.0a9
copier -f -r d3b4e70
pip install -r tooling/requirements_cicd.txt
flit install
