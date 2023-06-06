<#.SYNOPSIS
Reproduce the exact results during this commit.
#>

# Create the frozen virtual environment
$FROZEN_VENV = '.venv-frozen'
if ($Env:VIRTUAL_ENV) { 'deactivate' }
if (Test-Path $FROZEN_VENV) { Remove-Item -Recurse -Force $FROZEN_VENV }
$GLOBAL_PYTHON = 'py -3.10'
try { Invoke-Expression $("$GLOBAL_PYTHON --version") }
catch [System.Management.Automation.CommandNotFoundException] {
    $GLOBAL_PYTHON = 'python3.10'
}
Invoke-Expression "$GLOBAL_PYTHON -m venv $FROZEN_VENV"

# Activate environment
$VENV_ACTIVATE_WINDOWS = "$FROZEN_VENV/Scripts/activate"
$VENV_ACTIVATE_UNIX = "$FROZEN_VENV/bin/Activate.ps1"
if ( Test-Path $VENV_ACTIVATE_WINDOWS ) { . $VENV_ACTIVATE_WINDOWS }
elseif ( Test-Path $VENV_ACTIVATE_UNIX ) { . $VENV_ACTIVATE_UNIX }
else {
    throw [System.Management.Automation.ItemNotFoundException] 'Could not find a virtual environment.'
}

# Install the package and all frozen requirements
pip install --no-deps '.'
pip install --requirement 'repro.txt'

# Reproduce the results using DVC
dvc repro
