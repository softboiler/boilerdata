from pathlib import Path

import dill  # noqa: S403  # Only unpickling an internal object.

from boilerdata.models.project import Project


def get_model_fun():
    model_file = Project.get_project().dirs.model_file
    return dill.loads(  # noqa: S301  # Only unpickling an internal object.
        Path(model_file).read_bytes()
    )
