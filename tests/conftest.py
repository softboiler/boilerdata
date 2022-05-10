from contextlib import contextmanager
from os import chdir, getcwd
from pathlib import Path
from pytest import fixture


@contextmanager
def working_directory(path: Path):
    original_working_directory = getcwd()
    try:
        chdir(path)
        yield
    finally:
        chdir(original_working_directory)


@fixture
def tmp_cwd(tmp_path):
    with working_directory(tmp_path):
        yield tmp_path
