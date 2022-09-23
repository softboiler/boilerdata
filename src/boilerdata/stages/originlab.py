import originpro as op

from boilerdata.origin import get_path, open_originlab


def main():
    with open_originlab("data/plotter/results.opju"):
        gp = op.find_graph("lit_")
        fig = gp.save_fig(get_path("data/plots", mkdirs=True), type="png", width=800)
        if not fig:
            raise RuntimeError("Failed to save figure.")


if __name__ == "__main__":
    main()
