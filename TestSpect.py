from SpectraPlotter import SpectaPlot

test1 = SpectaPlot(cluster_size=6, isomer=["book", "cage", "ring"], method="intra",
                   plot_sticks=True, plot_convolutions=True)
test2 = SpectaPlot(cluster_size=6, isomer="book",
                   plot_sticks=True, plot_convolutions=True)
test2.makePlot()

