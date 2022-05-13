"""Generate configs for trials given their old layout."""

from models.project import Project
from models.trials import Coupon, Rod, Sample

from boilerdata.utils import load_config


def main():
    config, _ = load_config("project/config/project.toml", Project)
    good_trials = (config.trials / "Boiling Curves").iterdir()
    okay_trials = (config.trials / "Test Runs").iterdir()
    for trial in good_trials:
        match trial.name.split(" "):
            case [date, "Copper", spec]:
                date = date.replace(".", "-")
                match spec.split("-"):
                    case [rod, coupon, sample, _]:
                        rod = Rod(rod)
                        sample = Sample(sample)
                    case [rod, coupon, _]:
                        rod = Rod(rod)
                        sample = (
                            Sample.NA
                        )  # TODO: Make sure to get the Enum itself not the string, due to GetNameEnum
                    case [coupon, _]:
                        rod = Rod.W
                        sample = Sample.NA
                    case _:
                        raise ValueError(f'Could\'nt parse "{spec}"')
                coupon = Coupon(coupon)
                ...
            case _:
                pass
        ...
    ...


if __name__ == "__main__":
    main()
