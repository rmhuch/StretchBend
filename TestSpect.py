from SpectraPlotter import SpectaPlot

test1 = SpectaPlot(cluster_size=4, isomer=["3_1", "cage", "ring"], transition="SB", method="nm",
                   plot_sticks=True, plot_convolutions=True, delta=15)
test2 = SpectaPlot(cluster_size=4, isomer="ring",
                   plot_sticks=True, plot_convolutions=True)
test1.makePlot()

