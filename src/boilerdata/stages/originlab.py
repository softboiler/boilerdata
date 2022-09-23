from contextlib import contextmanager
from pathlib import Path

import originpro as op


def main():
    with open_originlab("data/plotter/results.opju"):
        gp = op.find_graph("lit_")
        fig = gp.save_fig(get_path("data/plots", mkdirs=True), type="png", width=800)
        if not fig:
            raise RuntimeError("Failed to save figure.")


@contextmanager
def open_originlab(file, readonly=True):
    """Open an OriginLab file."""
    if not Path(file).exists():
        raise FileNotFoundError(f"File not found: {file}")
    op.set_show(True)  # required
    yield op.open(file=get_path(file), readonly=readonly)
    op.exit()


def get_path(file, mkdirs=False):
    """Return the absolute path of a file for OriginLab interoperation."""
    path = Path(file)
    if mkdirs:
        path.mkdir(parents=True, exist_ok=True)
    return str(Path(file).resolve())


if __name__ == "__main__":
    main()
