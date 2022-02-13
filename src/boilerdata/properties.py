"""Get material properties."""

import os
import subprocess  # noqa: S404  # only used for hardcoded calls
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from time import sleep

import __main__
import numpy as np


def get_thermal_conductivity(
    material: str, temperatures, workdir: os.PathLike, ees: os.PathLike, wait: float = 7
):
    """Get thermal conductivity."""

    get_thermal_conductivity_script = dedent(
        """\
    $Import 'in.dat' Material$ N T[1..N]

    Duplicate j=1,N
        k[j] = Conductivity(Material$, T=T[j])
    End

    $Export 'out.dat' k[1..N]"""
    )

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

        script = Path("get_thermal_conductivity.txt")

        with open(script, "w+") as f:
            print(get_thermal_conductivity_script, file=f)

        # write post material, number of runs, and average post temperatures to in.dat
        with open("in.dat", "w+") as f:
            print(material, len(temperatures), *temperatures, file=f)
        # Invoke EES to write thermal conductivities to out.dat given contents of in.dat
        subprocess.Popen(  # noqa: S603, S607  # hardcoded
            [
                "pwsh",
                "-Command",
                f"{ees}",
                f"{script.resolve()}",
                "/solve",
            ]
        )
        sleep(wait)  # Wait long enough for EES to finish
        # EES should have written to out.dat
        with open("out.dat", "r") as f:
            k_str = f.read().split("\t")
            thermal_conductivity = np.array(k_str, dtype=np.float64)

    return thermal_conductivity
