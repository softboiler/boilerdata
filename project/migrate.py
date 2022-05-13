"""Generate configs for trials given their old layout."""

from bisect import insort
from pathlib import Path

from models.project import Project
from pydantic import BaseModel
import toml

from boilerdata.utils import load_config


class PartialTrial(BaseModel):
    date: str
    rod: str
    coupon: str
    sample: str
    group: str
    monotonic: bool
    joint: str
    comment: str


class PartialTrials(BaseModel):
    trials: list[PartialTrial]


def main():
    config, _ = load_config("project/config/project.toml", Project)
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
        date = date.replace(".", "-")

        trial = PartialTrial(
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

    Path("project/config/trials_raw.toml").write_text(
        toml.dumps(PartialTrials(trials=trials).dict())
    )


def match_sample_spec(spec):
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


if __name__ == "__main__":
    main()
