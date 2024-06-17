"""Generate docs."""

from pathlib import Path

import graphviz


def main():  # noqa: D103
    common = Path("docs/_static")
    dot = (common / "dag.dot").read_text(encoding="utf-8")
    g = graphviz.Source(dot)
    g.render(common / "dag", format="png", cleanup=True)


if __name__ == "__main__":
    main()
