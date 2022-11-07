"""A setup stage that should be run before DVC pipeline reproduction."""

from ruamel.yaml import YAML

from boilerdata.models.dirs import Dirs

yaml = YAML()
yaml.indent(offset=2)


def main():
    dirs = Dirs()
    proj = yaml.load(dirs.file_proj)
    proj["dirs"] = {
        k: str(v).replace("\\", "/") for k, v in dirs.dict(exclude_none=True).items()
    }
    yaml.dump(proj, dirs.file_proj)


if __name__ == "__main__":
    main()
