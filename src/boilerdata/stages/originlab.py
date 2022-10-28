from contextlib import contextmanager
from pathlib import Path
from time import sleep

import originpro as op

from boilerdata.models.project import Project


def main(proj: Project):
    with open_originlab(proj.dirs.originlab_results_file):
        gp = op.find_graph(proj.params.plots[0])
        fig = gp.save_fig(get_path(proj.dirs.plots, mkdirs=True), type="png", width=800)
        if not fig:
            raise RuntimeError("Failed to save figure.")


@contextmanager
def open_originlab(file, readonly=True):
    """Open an OriginLab file."""
    if not Path(file).exists():
        raise FileNotFoundError(f"File not found: {file}")
    op.set_show(True)  # required
    file = op.open(file=get_path(file), readonly=readonly)
    sleep(5)  # wait for data sources to update upon book opening
    yield file
    op.exit()


def get_path(file, mkdirs=False):
    """Return the absolute path of a file for OriginLab interoperation."""
    path = Path(file)
    if mkdirs:
        path.mkdir(parents=True, exist_ok=True)
    return str(Path(file).resolve())


if __name__ == "__main__":
    main(Project.get_project())
