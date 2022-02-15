"""Get material properties."""

from subprocess import run  # noqa: S404  # only used for hardcoded calls
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path

import numpy as np

EES_ROOT = Path("C:/EES32")
EES_PATH = EES_ROOT / "EES.exe"
LOOKUP_TABLES_PATH = EES_ROOT / "Userlib/EES_System/Incompressible"

# * -------------------------------------------------------------------------------- * #
# * MATERIAL PROPERTIES


def convert_lookup_tables(directory: Path):
    for table in get_lookup_tables():
        new_table = directory / table.relative_to(LOOKUP_TABLES_PATH).with_suffix(
            ".xlsx"
        )
        new_table.parent.mkdir(parents=True, exist_ok=True)
        text = (
            f"$OPENLOOKUP '{table.resolve()}' Lookup$\n"
            f"$SAVELOOKUP Lookup$ '{new_table.resolve()}'\n"
        )
        run_ees_script(Path(NamedTemporaryFile().name), text)


def get_thermal_conductivity(material: str, temperatures):
    """Get thermal conductivity."""

    with TemporaryDirectory() as tempdir:

        # Prepare input and output files inside of temporary directory
        files = {key: Path(tempdir) / f"{key}" for key in ["in", "out"]}

        # Write down material, number of runs, and temperatures
        files["in"].write_text(
            f"{material} {len(temperatures)} {' '.join([str(t) for t in temperatures])}"
        )

        # Run an EES script to find thermal conductivity and write out results
        run_ees_script(
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


# * -------------------------------------------------------------------------------- * #
# * HELPER FUNCTIONS


def get_materials():
    """Get all materials."""
    return (lkt.stem for lkt in get_lookup_tables())


def get_lookup_tables():
    """Get all lookup tables."""
    return LOOKUP_TABLES_PATH.rglob("*.lkt")


def run_ees_script(text: str):
    """Run an EES script."""
    with TemporaryDirectory() as tempdir:
        file = Path(tempdir) / "file"
        file.write_text(text)
        run([f"{EES_PATH}", f"{file.resolve()}", "/solve"])  # noqa: S603
