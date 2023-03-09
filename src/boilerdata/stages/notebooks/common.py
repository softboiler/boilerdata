import pandas as pd


def set_format():
    """Set up formatting for interactive notebook sessions.

    The triple curly braces in the f-string allows the format function to be dynamically
    specified by a given float specification. The intent is clearer this way, and may be
    extended in the future by making `float_spec` a parameter.
    """
    float_spec = ":#.4g"
    pd.options.display.min_rows = pd.options.display.max_rows = 50
    pd.options.display.float_format = f"{{{float_spec}}}".format
