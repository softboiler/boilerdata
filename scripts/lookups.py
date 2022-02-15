from pathlib import Path

from boilerdata import lookup, materials

WORKDIR = Path.home() / "Desktop/Lookup"
XLSX = WORKDIR / "XLSX"
XLSX_TWEAKED = WORKDIR / "XLSX Tweaked"
XLSX_FLATTENED = WORKDIR / "XLSX Flattened"
FEATHER = WORKDIR / "Feather"

lookup.get_xlsx_as_feather(XLSX_FLATTENED, FEATHER)
