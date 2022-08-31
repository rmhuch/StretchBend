import os
import numpy as np
import matplotlib.pyplot as plt
from McUtils.Plots import ListContourPlot

# TODO: write functions into 'BuildIntensityClusters' classes to interface with these functions

def plot_potential(dataDict):
    plt.rcParams.update({'font.size': 20})
    x = np.unique(dataDict["xyData"][:, 0])
    y = np.unique(dataDict["xyData"][:, 1])
    ens = dataDict["Energies"]
    ens[ens > 9000] = 9000
    pot = ens.reshape((len(x), len(y)))
    fig, ax = plt.subplots(figsize=(12, 12), dpi=216)
    ax.contourf(x, y*2, pot, levels=15, cmap="RdYlBu")
    CS = ax.contour(x, y*2, pot, levels=15, colors="black")
    ax.clabel(CS, inline=1, fmt='%1.0f', fontsize=16)
    ax.set_title(dataDict["Molecule"] + " " + dataDict["ScanCoords"] + " Potential Energy")
    data_name = dataDict["DataName"]
    plt.savefig(os.path.join(dataDict["MainDir"], "Figures", f"{data_name}_PE.png"), dpi=fig.dpi, bboxinches="tight")
    plt.close()

def plot_nonSquare(x, y, z):
    opts = dict(
        plot_style=dict(cmap="viridis_r", levels=15),
        axes_labels=['$\mathrm{R_{OH}}$ ($\mathrm{\AA}$)',
                     'HOH (Degrees)'])
    obj = ListContourPlot(np.column_stack((x, y, z)), **opts)
    obj.colorbar = {"graphics": obj.graphics}
    obj.show()

def plot_dipoles(dataDict):
    dips = dataDict["RotatedDipoles"]
    x = np.unique(dataDict["xyData"][:, 0])
    y = np.unique(dataDict["xyData"][:, 1])
    min_arg = np.argmin(dataDict["Energies"])
    print(min_arg)
    eq_coords = dataDict["xyData"][min_arg]
    comp = ['X', 'Y', 'Z']
    for i in np.arange(dips.shape[1]):
        if i == 0:
            plt.rcParams.update({'font.size': 20})
            shift_dips = dips[:, i] - dips[min_arg, i]
            dipoles = shift_dips.reshape((len(x), len(y)))
            fig, ax = plt.subplots(figsize=(14, 12), dpi=216)
            v = np.linspace(-3.0, 3.0, 21, endpoint=True)
            im = ax.contourf(x, y * 2, dipoles, v, cmap="RdYlBu", vmin=-3.0, vmax=3.0)
            plt.plot(eq_coords[0], eq_coords[1] * 2, marker="X", markersize=16, color="k")
            CS = ax.contour(x, y * 2, dipoles, v, colors="black")
            ax.clabel(CS, inline=1, fmt='%1.2f', fontsize=16)
            fig.colorbar(im, ticks=np.linspace(-3.0, 3.0, 7, endpoint=True), label="Debye")
            ax.set_title(dataDict["Molecule"] + " " + dataDict["ScanCoords"] + f" {comp[i]} Dipole")
            data_name = dataDict["DataName"]
            plt.savefig(os.path.join(dataDict["MainDir"], "Figures", f"{data_name}_{comp[i]}dipoles.png"), dpi=fig.dpi,
                        bboxinches="tight")
            plt.close()
        else:
            pass

def plot_dipoleswWFNS(dataDict):
    data_name = dataDict["DataName"]
    dips = dataDict["RotatedDipoles"]
    x = np.unique(dataDict["xyData"][:, 0])
    y = np.unique(dataDict["xyData"][:, 1])
    min_arg = np.argmin(dataDict["Energies"])
    print(dataDict["xyData"][min_arg])
    eq_coords = dataDict["xyData"][min_arg]
    comp = ['X', 'Y', 'Z']
    # pull ground state wave function
    DVRdat = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_2D_DVR.npz"))
    gs_wfn = DVRdat["wfns_array"][:, 0]
    for i in np.arange(dips.shape[1]):
        if i == 0:
            plt.rcParams.update({'font.size': 20})
            shift_dips = dips[:, i] - dips[min_arg, i]
            dipoles = shift_dips.reshape((len(x), len(y)))
            fig, ax = plt.subplots(figsize=(14, 12), dpi=216)
            v = np.linspace(-3.0, 3.0, 21, endpoint=True)
            # plot dipole contour
            im = ax.contourf(x, y * 2, dipoles, v, cmap="RdYlBu", vmin=-3.0, vmax=3.0)  # , extend="both")
            # plot eq point
            plt.plot(eq_coords[0], eq_coords[1] * 2, marker="X", markersize=16, color="k")
            # plot wave function
            ax.contour(x, y * 2, gs_wfn.reshape((len(x), len(y))).T, colors="black")
            fig.colorbar(im, ticks=np.linspace(-3.0, 3.0, 7, endpoint=True), label="Debye")
            ax.set_title(dataDict["Molecule"] + " " + dataDict["ScanCoords"] + f" {comp[i]} Dipole")
            data_name = dataDict["DataName"]
            plt.savefig(os.path.join(dataDict["MainDir"], "Figures", f"{data_name}_{comp[i]}dipoleswWFNS.png"), dpi=fig.dpi,
                        bboxinches="tight")
            plt.close()
        else:
            pass


