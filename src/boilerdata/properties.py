"""Get material properties."""

import os
import subprocess  # noqa: S404  # only used for hardcoded calls
from contextlib import contextmanager
from pathlib import Path
from time import sleep

import numpy as np

import __main__


def get_thermal_conductivity(
    material: str, temperatures, wait: float, workdir: os.PathLike, ees: os.PathLike
):
    """Get thermal conductivity."""

    @contextmanager
    def change_directory(path: os.PathLike):
        """Context manager for changing working directory."""
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    with change_directory(workdir):

        # write post material, number of runs, and average post temperatures to in.dat
        with open("in.dat", "w+") as f:
            print(material, len(temperatures), *temperatures, file=f)
        # Invoke EES to write thermal conductivities to out.dat given contents of in.dat
        subprocess.Popen(  # noqa: S603, S607  # hardcoded
            [
                "pwsh",
                "-Command",
                f"{ees}",
                f"{Path('get_thermal_conductivity.ees').resolve()}",
                "/solve",
            ]
        )
        sleep(wait)  # Wait long enough for EES to finish
        # EES should have written to out.dat
        with open("out.dat", "r") as f:
            k_str = f.read().split("\t")
            thermal_conductivity = np.array(k_str, dtype=np.float64)

    return thermal_conductivity
