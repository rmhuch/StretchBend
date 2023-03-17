from SpectraPlotter import SpectraPlot

coup2 = ["lm", "nm"]
coup4M = ["lm", "OH", "bnd", "nm"]
coup4I = ["lm", "HOH", "intra", "nm"]
coup5 = ["lm", "OH", "HOH", "intra", "nm"]
coup6 = ["lm", "OH", "bnd", "HOH", "intra", "nm"]
test22 = SpectraPlot(cluster_size=4, isomer=["cage", "ring", "3_1"], coupling=coup6, transition="SB",
                     plot_sticks=True, plot_convolutions=True, delta=10)
test22.plot_Spect()

