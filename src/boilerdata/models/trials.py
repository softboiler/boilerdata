import datetime
import re
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field, FilePath, validator

from boilerdata.models.axes_enum import AxesEnum as A  # noqa: N814
from boilerdata.models.common import MyBaseModel
from boilerdata.models.dirs import Dirs
from boilerdata.models.enums import Coupon, Group, Joint, Rod, Sample
from boilerdata.models.geometry import Geometry


class Trial(MyBaseModel):
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
    sample: Optional[Sample]
    joint: Joint
    sixth_tc: bool = Field(
        default=False,
        description="Whether this trial includes a thermocouple at the top of the coupon.",
    )
    good: bool = Field(
        default=True,
        description="Whether the boiling curve is good.",
    )
    new: bool = Field(
        default=False,
        description="Whether this is newly-collected data.",
    )

    # ! FIELDS TO EXCLUDE FROM DATAFRAME

    # Named "trial" as in "the date this trial was run".
    @property
    def trial(self):
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

    def setup(self, dirs: Dirs, geometry: Geometry):
        self.set_paths(dirs)
        self.set_geometry(geometry)

    def set_paths(self, dirs: Dirs):
        """Get the path to the data for this trial. Called during project setup."""
        trial_base = dirs.trials / str(self.trial.date())
        self.path = trial_base / dirs.per_trial if dirs.per_trial else trial_base
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

            trial_date = self.trial.isoformat()  # for consistency across datetimes
            run_index.append(
                tuple(
                    pd.Timestamp.fromisoformat(item) for item in [trial_date, run_time]
                )
            )
        self.run_index = run_index

    def set_geometry(self, geometry: Geometry):
        """Get relevant geometry for the trial."""
        thermocouples = [A.T_1, A.T_2, A.T_3, A.T_4, A.T_5]
        thermocouple_pos = geometry.rods[self.rod] + geometry.coupons[self.coupon]  # type: ignore  # due to use_enum_values
        if self.sixth_tc:
            thermocouples.append(A.T_6)
            # Since zero is defined as the surface
            thermocouple_pos = np.append(thermocouple_pos, 0)
        self.thermocouple_pos = dict(zip(thermocouples, thermocouple_pos))


class Trials(MyBaseModel):
    """The trials."""

    trials: list[Trial]
