import datetime
import re
from pathlib import Path

import pandas as pd
from pydantic import DirectoryPath, Field, FilePath, validator

from boilerdata.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models import ProjectModel, YamlModel
from boilerdata.models.enums import Coupon, Group, Joint, Rod, Sample
from boilerdata.models.geometry import Geometry
from boilerdata.models.paths import Paths


class Trial(ProjectModel):
    """A trial."""

    # ! META FIELDS ADDED TO DATAFRAME

    # !! DATE

    date: datetime.date = Field(
        default=...,
        description="The date of the trial.",
        exclude=True,
    )

    @validator("date", pre=True)
    def validate_date(cls, date):
        return datetime.date.fromisoformat(date)

    group: Group
    rod: Rod
    coupon: Coupon
    sample: Sample | None
    joint: Joint
    good: bool = Field(
        default=True,
        description="Whether the boiling curve is good.",
    )
    new: bool = Field(
        default=False,
        description="Whether this is newly-collected data.",
    )

    # ! FIELDS TO EXCLUDE FROM DATAFRAME

    @property
    def timestamp(self):
        return pd.Timestamp(self.date)

    # Loaded from config, but not propagated to dataframes. Not readable in a table
    # anyways, and NA-handling results in additional ~40s to pipeline due to the need to
    # use slow "fillna".
    comment: str = Field(
        default="",
        exclude=True,
    )

    # ! PROJECT-DEPENDENT SETUP

    # Can't be None. Set in Project.__init__()
    path: DirectoryPath = Field(
        default=None,
        exclude=True,
    )
    run_files: list[FilePath] = Field(
        default=None,
        exclude=True,
    )
    run_index: list[tuple[pd.Timestamp, pd.Timestamp]] = Field(
        default=None,
        exclude=True,
    )
    thermocouple_pos: dict[str, float] = Field(
        default=None,
        exclude=True,
    )

    def setup(self, paths: Paths, geometry: Geometry, copper_temps: list[A]):
        self.set_paths(paths)
        self.set_geometry(geometry, copper_temps)

    def set_paths(self, paths: Paths):
        """Get the path to the data for this trial. Called during project setup."""
        trial_base = paths.trials / self.date.isoformat()
        self.path = trial_base
        self.run_files = sorted(self.path.glob("*.csv"))
        if not self.run_files:
            raise FileNotFoundError(f"No runs found in {self.path}.")
        self.set_index()

    def set_index(self):
        """Get the multiindex for all runs. Called during project setup."""
        run_re = re.compile(r"(?P<date>.*)T(?P<time>.*)")
        run_index: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for run_file in self.run_files:
            run_time = run_file.stem.removeprefix("results_")

            if m := run_re.match(run_time):
                run_time = f"{m['date']}T{m['time'].replace('-', ':')}"
            else:
                raise AttributeError(f"Could not parse run time: {run_time}")

            trial_date = self.date.isoformat()  # for input to fromisoformat() below
            run_index.append(
                tuple(
                    pd.Timestamp.fromisoformat(item) for item in [trial_date, run_time]
                )
            )
        self.run_index = run_index

    def set_geometry(self, geometry: Geometry, copper_temps: list[A]):
        """Get relevant geometry for the trial."""
        thermocouple_pos = geometry.rods[self.rod] + geometry.coupons[self.coupon]  # type: ignore  # pyright: 1.1.318, pydantic: use_enum_values
        self.thermocouple_pos = dict(zip(copper_temps, thermocouple_pos, strict=True))  # type: ignore  # pyright: 1.1.310, pydantic: use_enum_values


class Trials(YamlModel):
    """The trials."""

    trials: list[Trial]

    def __init__(self, data_file: Path):
        super().__init__(data_file)
