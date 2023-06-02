from SpectraPlotter import SpectraPlot

coup2 = ["lm", "bnd"]
coup4M = ["lm", "OH", "bnd", "nm"]
coupMS = ["lm", "HOH", "nm"]  # plots used in SB2 MS
coup5 = ["lm", "OH", "HOH", "intra", "nm"]
coup6 = ["lm", "OH", "bnd", "HOH", "intra", "nm"]

test22 = SpectraPlot(cluster_size=6, isomer=["cage"], coupling=["nm"], transition="SB",
                     plot_sticks=True, plot_convolutions=True, delta=10)
test22.plot_Spect()

