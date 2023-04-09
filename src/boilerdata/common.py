from os import PathLike
from pathlib import Path

StrPath = str | PathLike[str]


def get_file(path: StrPath, create: bool = False) -> Path:
    """Generate `pathlib.Path` to a file that exists.

    Handle the "~" user construction if necessary and return a `pathlib.Path` object.
    Raise exception if the file is not found.

    Args:
        path: Path.
        create: Whether a file should be created if it doesn't already exist.

    Returns:
        pathlib.Path: The path after handling.

    Raises:
        FleNotFoundError: If the file doesn't exist or does not refer to a file.
    """
    path = expanduser2(path) if isinstance(path, str) else Path(path)
    if not path.exists():
        if create:
            path.touch()
        else:
            raise FileNotFoundError(f"The path '{path}' does not exist.")
    elif not path.is_file():
        raise FileNotFoundError(f"The path '{path}' does not refer to a file.")
    return path


def expanduser2(path: StrPath) -> Path:
    """Expand the "~" user construction.

    Unlike the builtin `posixpath.expanduser`, this always works on Windows, and returns
    a `pathlib.Path` object.

    Args:
        path: A string that may contain "~" at the start.

    Returns:
        pathlib.Path: The path after user expansion.
    """
    home = "~/"
    if isinstance(path, str) and path.startswith(home):
        return Path.home() / path.lstrip(home)
    else:
        return Path(path)
