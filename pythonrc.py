"""Startup for IPython and the REPL. Isn't run for notebooks (see `ipythonrc.py`).

Avoid activating Rich features that break functionality outside of the REPL.
"""


def main():

    from rich import (
        inspect,  # pyright: ignore [reportUnusedImport]  # For interactive mode
        traceback,
    )

    traceback.install()

    if not is_notebook_or_ipython():
        from rich import (
            pretty,
            print,  # pyright: ignore [reportUnusedImport]  # For interactive mode
        )

        pretty.install()


# https://stackoverflow.com/a/39662359
def is_notebook_or_ipython() -> bool:
    try:
        shell = (
            get_ipython().__class__.__name__  # pyright: ignore [reportUndefinedVariable]  # Dynamic
        )
        return shell == "TerminalInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


main()
