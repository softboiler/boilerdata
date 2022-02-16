"""Get material properties."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from boilerdata import ees


def get_thermal_conductivity_old(material: str, temperatures):
    """Get thermal conductivity."""

    with TemporaryDirectory() as tempdir:

        # Prepare input and output files inside of temporary directory
        files = {key: Path(tempdir) / f"{key}" for key in ["in", "out"]}

        # Write down material, number of runs, and temperatures
        files["in"].write_text(
            f"{material} {len(temperatures)} {' '.join([str(t) for t in temperatures])}"
        )

        # Run an EES script to find thermal conductivity and write out results
        ees.run_script(
            f"$Import '{files['in'].resolve()}' Material$ N T[1..N]\n"
            "\n"
            "Duplicate j=1,N\n"
            "    k[j] = Conductivity(Material$, T=T[j])\n"
            "End\n"
            "\n"
            f"$Export '{files['out'].resolve()}' k[1..N]\n"
        )

        # Clean up results from EES and convert to floats
        return np.array(
            [np.float64(val) for val in files["out"].read_text().strip().split("\t")]
        )
