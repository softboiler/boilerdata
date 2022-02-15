"""Operate on lookup tables, including generation, tweaking, and conversion."""

from pathlib import Path

from boilerdata import ees
from boilerdata.config import settings

EES_LOOKUP_TABLES_PATH = (
    settings.ees_root / "Userlib/EES_System/Incompressible"  # type: ignore
)


def get_lookup_tables(destination_directory: Path):
    """Put all lookup tables in the destination in XLSX format."""
    # E to XLSX not CSV, so that units are written by EES.
    for table in get_lookup_table_paths():
        new_table = destination_directory / table.relative_to(
            EES_LOOKUP_TABLES_PATH
        ).with_suffix(".xlsx")
        new_table.parent.mkdir(parents=True, exist_ok=True)
        ees.run_script(
            f"$OPENLOOKUP '{table.resolve()}' Lookup$\n"
            f"$SAVELOOKUP Lookup$ '{new_table.resolve()}'\n"
        )


def get_materials():
    """Get all materials."""
    return (lkt.stem for lkt in get_lookup_table_paths())


def get_lookup_table_paths():
    """Get all lookup tables."""
    return EES_LOOKUP_TABLES_PATH.rglob("*.lkt")
