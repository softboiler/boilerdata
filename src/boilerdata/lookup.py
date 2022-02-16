"""Operate on lookup tables, including generation, tweaking, and conversion."""

from pathlib import Path
import re
import shutil

from boilerdata import ees
from boilerdata.config import settings
import pandas as pd

EES_LOOKUP_TABLES_PATH = (
    settings.ees_root / "Userlib/EES_System/Incompressible"  # type: ignore
)


def get_xlsx_conductivity_as_feather(
    source_directory: Path, destination_directory: Path
):
    destination_directory.mkdir(parents=False, exist_ok=True)
    for table in sorted(source_directory.glob("*.xlsx")):
        try:
            df = pd.read_excel(
                table, sheet_name="EES data", skiprows=[1], usecols=["T", "k"]
            )
        except ValueError:
            continue
        df.dropna().reset_index(drop=True).to_feather(
            destination_directory / f"{table.stem}.feather"
        )


def flatten_xlsx(source_directory: Path, destination_directory: Path):
    """Flatten the XLSX directory structure into a single directory."""
    destination_directory.mkdir(parents=False, exist_ok=True)
    for source_table in sorted(source_directory.rglob("*.xlsx")):
        destination_table = destination_directory / source_table.name
        shutil.copy(source_table, destination_table)


def tweak_xlsx(source_directory: Path, destination_directory: Path):
    """Copy files to new destination, sanitized and all uppercase."""
    destination_directory.mkdir(parents=False, exist_ok=True)
    source_tables_processed: list[Path] = []
    for source_table in sorted(source_directory.rglob("*.xlsx")):
        if any(source_table.stem == table.stem for table in source_tables_processed):
            continue
        clean_stem = source_table.stem.upper().replace(" ", "")
        starts_with_digit = re.compile(r"^\d")
        if starts_with_digit.match(clean_stem):
            clean_stem = f"M{clean_stem}"
        destination_table = (
            destination_directory
            / source_table.parent.relative_to(source_directory)
            / f"{re.sub(pattern=r'[()-.#]', repl='_', string=clean_stem)}.xlsx"
        )
        destination_table.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_table, destination_table)
        source_tables_processed.append(source_table)


def get_all_lkt_as_xlsx(destination_directory: Path):
    """Put all lookup tables in the destination in XLSX format."""
    for table in get_lkt_paths():
        get_lkt_as_xlsx(destination_directory, table)


def get_lkt_as_xlsx(destination_directory: Path, lkt_table: Path):
    """Put a lookup table in the destination in XLSX format."""
    # Save to XLSX, not CSV, so that units are written properly by EES.
    new_table = destination_directory / lkt_table.relative_to(
        EES_LOOKUP_TABLES_PATH
    ).with_suffix(".xlsx")
    new_table.parent.mkdir(parents=True, exist_ok=True)
    ees.run_script(
        f"$OPENLOOKUP '{lkt_table.resolve()}' Lookup$\n"
        f"$SAVELOOKUP Lookup$ '{new_table.resolve()}'\n"
    )


def get_lkt_paths():
    """Get all lookup tables."""
    return EES_LOOKUP_TABLES_PATH.rglob("*.lkt")
