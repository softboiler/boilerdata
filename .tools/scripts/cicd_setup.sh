pip install -U pip setuptools wheel
pip install -r .tools/requirements/requirements_cicd.txt
pip install .
echo "I AM ABOUT TO INSTALL DYNAMIC REQUIREMENTS"
pip install -r requirements_dynamic.txt
echo "I HAVE FINISHED INSTALLING DYNAMIC REQUIREMENTS"
