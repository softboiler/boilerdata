"""Pipe fittings for inspecting `pandas` pipelines.

Pandas allows method chaining of user-supplied functions via `pipe`. This module
facilitates pipeline inspection either by tapping into a function that you control via
the `tap` decorator, or by inserting a `tee` into the pipeline as in
`df.<pipeline>.pipe(tee).<pipeline>`.
"""

import logging
from functools import wraps
from typing import Callable

import pandas as pd

pdobj = pd.DataFrame | pd.Series


def default_preview(df: pdobj) -> str:
    """Default preview function for a `pandas` dataframe or series."""
    return (
        f"type: {type(df)}"
        f"\nshape: {df.shape}"
        f"\nstats:\n{df.describe(percentiles=[])}"
    )


def tap(enable: bool = True, preview: Callable[[pdobj], str] = default_preview):
    """Decorate a function to tap into a `pandas` pipeline and preview the dataframe.

    Pandas allows method chaining of user-supplied functions via `pipe`. When this
    decorator adorns such a user-supplied function, it will log the function name, the
    keyword arguments passed to the function, and a preview of the resulting dataframe
    or series (by default: its type, shape, and statistics) to the `INFO` log level.

    A custom `preview` function may also be provided, which must take a dataframe or
    series and return a string.

    Parameters
    ----------
    enable : bool, optional
        Enable the tap. Default is True.
    preview: Callable[[pandas.DataFrame | pandas.Series], str], optional
        The preview function. Default previews its type, shape, and statistics.

    Example
    -------
    import pandas as pd
    import numpy as np

    def main():
        pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        df = df.<pipeline>.pipe(my_func).<pipeline>

    @tap(enable=True)
    def my_func(df):
        # Arbitrary implementation
        return df

    if __name__ == "__main__":
        main()
    """

    def decorator(func):  # type: ignore
        @wraps(func)
        def wrapper(df: pdobj, **kwargs) -> pdobj:  # type: ignore

            if enable:
                df = func(df, **kwargs)
                logging.info(
                    f"\nfunc: {func.__name__}"
                    f"\nkwargs: {kwargs}"
                    f"\n{preview(df)}"
                    "\n"
                )
            else:
                df = func(df, **kwargs)
            return df

        return wrapper

    return decorator


def tee(
    df: pdobj, enable: bool = True, preview: Callable[[pdobj], str] = default_preview
) -> pdobj:
    """Insert into a `pandas` pipeline e.g. `df.pipe(tee)` and preview the dataframe.

    Pandas allows method chaining of user-supplied functions via `pipe`. When this
    function is part of the pipeline, it will log a preview of the resulting dataframe
    (its type, shape, and statistics) to the `INFO` log level.

    A custom `preview` function may also be provided, which must take a dataframe or
    series and return a string.

    Parameters
    ----------
    df : pandas.DataFrame | pandas.Series
        A `pandas` dataframe or series.
    enable : bool, optional
        Enable the tee. Default is True.

    Example
    --------
    import pandas as pd
    import numpy as np

    def main():
        pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        df = df.<pipeline>.pipe(tee).<pipeline>

    if __name__ == "__main__":
        main()
    """
    if enable:
        logging.info(f"\nfunc: tee\n{preview(df)}\n")
    return df
