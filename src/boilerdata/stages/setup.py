"""A setup stage that should be run before DVC pipeline reproduction."""

# sourcery skip: name-type-suffix  # dirs_dict is fine

from pathlib import Path

from ruamel.yaml import YAML

from boilerdata.models.dirs import Dirs

yaml = YAML()
yaml.indent(offset=2)


def main():
    dirs = Dirs()
    proj = yaml.load(dirs.file_proj)
    dirs_dict = dirs.dict(exclude_none=True)
    proj["dirs"] = repl_path(dirs_dict)
    proj["dirs"]["originlab_plot_files"] = repl_path(dirs_dict["originlab_plot_files"])
    yaml.dump(proj, dirs.file_proj)


def repl_path(dirs_dict: dict[str, Path]):
    """Replace Windows path separator with POSIX separator."""
    return {k: str(v).replace("\\", "/") for k, v in dirs_dict.items()}


if __name__ == "__main__":
    main()
