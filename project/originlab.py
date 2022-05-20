import pyperclip

from boilerdata.utils import load_config
from models import Columns

columns = load_config("project/config/columns.yaml", Columns)
pyperclip.copy(columns.generate_originlab_column_designation_string())
