from functools import (
    wraps,  # pyright: ignore [reportUnusedImport]  # Needed for unpickled model
)
from pathlib import Path
import warnings

import dill  # noqa: S403  # Only unpickling an internal object.
import numpy as np  # pyright: ignore [reportUnusedImport]  # Needed for unpickled model

from boilerdata.models.project import Project

model_file = Project.get_project().dirs.model_file
file_bytes = Path(model_file).read_bytes()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unpickled_model = dill.loads(file_bytes)  # noqa: S301  # Known unpickling.

model = unpickled_model.get_model()
