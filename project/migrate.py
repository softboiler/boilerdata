"""Generate configs for trials given their old layout."""

from bisect import insort
from collections import Counter
from operator import indexOf
from pathlib import Path
from textwrap import dedent

import pandas as pd
from pydantic import BaseModel, DirectoryPath

from boilerdata.utils import StrPath, dump_model, load_config
from models import Project
from pipeline import get_defaults, get_label, get_units


def main():
    project, _ = get_defaults()
    migrate_2(project, Path("project/columns.py"))


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
        labels = [get_label(label) for label in df.columns]
        units = [get_units(label) for label in df.columns]

        counter = Counter(labels).items()
        dupes = []
        for label, count in counter:
            if count > 2:
                raise NotImplementedError("Can't handle triplicates or higher.")
            elif count > 1:
                dupes.append(rindex(labels, label))

        for index in dupes:
            labels[index] += "_dupe"

        columns = [
            Column(label=label, units=unit) for label, unit in zip(labels, units)
        ]

        text = dedent(
            """\
            # flake8: noqa

            from enum import auto

            from boilerdata.enums import GetNameEnum


            class Columns(GetNameEnum):
            """
        )
        for column in columns:
            label = column.label.replace("\u2206", "D").replace("/", "_")
            text += f"    {label} = auto()\n"
        columns_path.write_text(text)

    class Column(BaseModel):
        """Configuration for a column after Migration 2."""

        label: str
        units: str

    # https://stackoverflow.com/a/63834895
    def rindex(lst, value):
        return len(lst) - indexOf(reversed(lst), value) - 1

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


if __name__ == "__main__":
    main()
