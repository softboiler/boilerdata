"""Generate configs for trials given their old layout."""

from bisect import insort
from collections import Counter
from operator import indexOf
from pathlib import Path
from textwrap import dedent

import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field

from boilerdata.utils import StrPath, dump_model, load_config, write_schema
from models import Project
from pipeline import get_defaults


def main():
    project, _ = get_defaults()
    migrate_3(
        project, "project/schema/columns_schema.json", "project/config/columns.yaml"
    )


def migrate_3(project: Project, columns_schema_path: StrPath, columns_path: StrPath):
    """Migration 3: Generate columns config.

    Parameters
    ----------
    project: Project
        The project model.
    columns_schema_path: StrPath
        The path to the columns schema.
    columns_path: StrPath
        The path to the columns configuration file.
    """

    def main():

        df = pd.read_csv(project.results_file, index_col=0)
        units = [get_units(column) for column in df.columns]

        names = dedupe_columns(df)
        columns = [Column(units=unit) for unit in units]
        columns = Columns(columns=dict(zip(names, columns)))

        write_schema(columns_schema_path, Columns)
        dump_model(columns_path, columns)

    def get_units(label: str) -> str:
        match label.split():
            case label, units:
                return units.strip("()")
            case _:
                return ""

    class Column(BaseModel):
        """Configuration for a column after Migration 3."""

        pretty_name: str = Field(
            default=None,
            description="The column name.",
        )
        units: str = Field(
            default=...,
            description="The units for this column's values.",
        )

    class Columns(BaseModel):
        """Configuration for a column after Migration 3."""

        columns: dict[str, Column]

    main()


def migrate_2(project: Project, columns_path: Path):
    """Migration 2: Generate columns enum.

    Parameters
    ----------
    project: Project
        The project model.
    columns_path: StrPath
        The path to `columns.py`.
    """

    def main():

        df = pd.read_csv(project.results_file, index_col=0)
        labels = dedupe_columns(df)

        text = dedent(
            """\
            # flake8: noqa

            from enum import auto

            from boilerdata.enums import GetNameEnum


            class Columns(GetNameEnum):
            """
        )
        for label in labels:
            text += f"    {label} = auto()\n"
        columns_path.write_text(text, encoding="utf-8")

    main()


def migrate_1(project_path: StrPath, trials_path: StrPath):
    """Migration 1: Partially populate a config file containing trials info.

    This migration was informed by the original trials folder structure. The "Boiling
    Curves" subfolder had mostly well-behaved, mostly monotonic results. The "Test Runs"
    subfolder had unbehaved trials, with less distinct folder names.

    Parameters
    ----------
    project_path: StrPath
        The path to `project.yaml`.
    trials_path: StrPath
        The path to `trials.yaml`.
    """

    def main():
        config = load_config(project_path, Project)
        good_trials = list((config.trials / "Boiling Curves").iterdir())
        okay_trials = list((config.trials / "Test Runs").iterdir())
        trials = []
        for trial in good_trials + okay_trials:

            match trial.name.split(" "):
                case [date, "Copper", spec] if trial in good_trials:
                    comment = ""
                    rod, coupon, sample, group = match_sample_spec(spec)
                case [date, *rest]:
                    comment = " ".join(rest)
                    rod = coupon = sample = group = ""
                case _:
                    raise ValueError(f'Couldn\'t parse "{trial.name}"')

            monotonic = trial in good_trials
            joint = ""
            date = "20" + date.replace(".", "-")

            trial = Trial(
                date=date,
                rod=rod,
                coupon=coupon,
                sample=sample,
                group=group,
                monotonic=monotonic,
                joint=joint,
                comment=comment,
            )

            insort(trials, trial, key=lambda t: t.date)

        dump_model(trials_path, Trials(trials=trials))

    def match_sample_spec(spec):
        """Match the dash-delimited sample specification of various formats."""

        match spec.split("-"):
            case [rod, coupon, sample, _]:  # Like X-A6-B3-1
                pass
            case [rod, coupon, _]:  # Like Y-A7-2
                sample = "NA"
            case [coupon, _]:  # Like A4-12
                rod = "W"
                sample = "NA"
            case _:
                raise ValueError(f'Couldn\'t parse "{spec}"')

        if "B" in sample:
            group = "porous"
        else:
            group = "control"

        return rod, coupon, sample, group

    class Project(BaseModel):
        """Project configuration prior to Migration 1."""

        trials: DirectoryPath

    class Trial(BaseModel):
        """Configuration for a single trial after Migration 1.

        This configuration is less strict because we aren't informing their values from
        specified Enums.
        """

        date: str
        rod: str
        coupon: str
        sample: str
        group: str
        monotonic: bool
        joint: str
        comment: str

    class Trials(BaseModel):
        """Top-level configuration for a list of trials after Migration 1."""

        trials: list[Trial]

    main()


# * -------------------------------------------------------------------------------- * #
# * COMMON FUNCTIONS


def dedupe_columns(df):
    labels = [
        label.split()[0].replace("\u2206", "D").replace("/", "_")
        for label in df.columns
    ]
    counter = Counter(labels).items()
    dupes = []
    for label, count in counter:
        if count > 2:
            raise NotImplementedError("Can't handle triplicates or higher.")
        elif count > 1:
            dupes.append(rindex(labels, label))

    for index in dupes:
        labels[index] += "_dupe"
    return labels


# https://stackoverflow.com/a/63834895
def rindex(lst, value):
    return len(lst) - indexOf(reversed(lst), value) - 1


# * -------------------------------------------------------------------------------- * #
# * MAIN


if __name__ == "__main__":
    main()
