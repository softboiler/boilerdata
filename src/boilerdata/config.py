from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="BOILERDATA",
    settings_files=["settings.toml"],
)
