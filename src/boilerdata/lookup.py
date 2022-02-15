"""Operate on lookup tables, including generation, tweaking, and conversion."""

from pathlib import Path
from boilerdata import ees

LOOKUP_TABLES_PATH = ees.EES_ROOT / "Userlib/EES_System/Incompressible"


def convert_lookup_tables(directory: Path):
    for table in get_lookup_tables():
        new_table = directory / table.relative_to(LOOKUP_TABLES_PATH).with_suffix(
            ".xlsx"
        )
        new_table.parent.mkdir(parents=True, exist_ok=True)
        ees.run_script(
            f"$OPENLOOKUP '{table.resolve()}' Lookup$\n"
            f"$SAVELOOKUP Lookup$ '{new_table.resolve()}'\n"
        )


def get_materials():
    """Get all materials."""
    return (lkt.stem for lkt in get_lookup_tables())


def get_lookup_tables():
    """Get all lookup tables."""
    return LOOKUP_TABLES_PATH.rglob("*.lkt")
