python -m pip install -U pip
pip install -U setuptools wheel
pip install -r .tools/requirements/requirements_cicd.txt
pip install .
pip install -r requirements_dynamic.txt
