import pyperclip

from utils import get_project

project = get_project()
pyperclip.copy(project.get_originlab_coldes())
