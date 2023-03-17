import numpy as np
import os
from ProtonatedHexSpectra import PlotVPTSpect

Delta = 10
for i in ["full", "Stretch"]:  # region options: full, Stretch, MJStretch (allD only), SB
    allH = PlotVPTSpect(isdueterated=False, anharm=False, mixData=True, basis="tz", region=i, delta=Delta,
                        isomer=["e2", "t2", "z1"], plot_exp=True)
    allD = PlotVPTSpect(isdueterated=True, anharm=False, mixData=True, plot_exp=True, basis="tz", region=i, delta=Delta,
                        isomer=["e2", "t2", "z1"])
    allH.plot_Spect()
    allD.plot_Spect()

