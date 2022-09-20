import originpro as op

from boilerdata.origin import get_path, open_originlab

with open_originlab("data/plotter/results.opju"):
    gp = op.find_graph("lit_")
    gp.save_fig(get_path("data/plots", mkdirs=True), width=800)
