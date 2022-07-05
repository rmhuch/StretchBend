import os
import glob
import matplotlib.cm
import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from Converter import Constants
from McUtils.GaussianInterface import GaussianLogReader
from McUtils.Plots import ListContourPlot
from Rotator import *

def pull_data(logfile_pattern, ens_wave=True):
    """ takes a list of log files and returns a dictionary filled with Gaussian data from the logfiles
        :param logfile_pattern, string pattern of log files (path pulled and globed in function)
        :param ens_wave - if True, energies are shifted so the minimum is @ 0 in cm^-1, if False, energies in Hartree
        :returns dictionary ('MainDir' - path to directory, 'Molecule' - shorthand of system (used in file naming),
                 'ScanCoords' - shorthand of scanned coordinates (used in file naming) 'AtomStr' - order of atoms,
                 'Dipoles' - dipoles from Gaussian (later rotated), 'xyData' - values of the scanned coordinates,
                 'Energies' - Gaussian energies, 'Cartesians' -  Coordinates at every scan point"""
    dataDict = dict()
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    # assign values for all data
    dataDict["MainDir"] = MainDir
    items = logfile_pattern.split("_")
    if items[0] == "w2":
        dataDict["Molecule"] = "D/A"
    elif items[0] == "w6":
        dataDict["Molecule"] = "AAD/ADD"
    elif items[0] == "w6a":
        dataDict["Molecule"] = "ADD/AAD"
    elif items[0] == "w1":
        dataDict["Molecule"] = "Monomer"
    else:
        raise Exception(f"can not define {items[0]} molecules")
    if dataDict["Molecule"] == "Monomer":
        dataDict["ScanCoords"] = "Free Hydrogen"
        dataDict["DataName"] = items[0] + "_RB"
    elif items[1].find("4") > 0:
        dataDict["ScanCoords"] = "Free Hydrogen"
        dataDict["DataName"] = items[0] + "_" + items[1][4:7]
    elif items[1].find("5") > 0:
        dataDict["ScanCoords"] = "H-Bound Hydrogen"
        dataDict["DataName"] = items[0] + "_" + items[1][4:7]
    if items[0] == "w2":
        dataDict["AtomStr"] = ["O", "H", "H", "O", "H", "H"]
    elif items[0] == "w1":
        dataDict["AtomStr"] = ["O", "H", "H"]
    elif items[0] == "w6" or items[0] == "w6a":
        dataDict["AtomStr"] = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    # parse through logfiles to collect the rest of the data
    dips = []
    xyData = []
    ens = []
    cats = []
    logfiless = glob.glob(os.path.join(MainDir, logfile_pattern))
    for logpath in logfiless:
        with GaussianLogReader(logpath) as reader:
            parse = reader.parse(("ScanEnergies", "DipoleMoments", "StandardCartesianCoordinates"))
        dips.append(list(parse["DipoleMoments"]))
        raw_data = parse["ScanEnergies"][1]
        if logpath[-5] is "a":
            xyData.append(np.column_stack((raw_data[:, 1], np.repeat(64.1512, len(raw_data[:, 1])))))
        else:
            xyData.append(np.column_stack((raw_data[:, 1], raw_data[:, 2])))
        ens.append(raw_data[:, -1])  # returns MP2 energy
        cats.append(parse["StandardCartesianCoordinates"][1])
    # concatenate lists
    dipoles = np.concatenate(dips)
    scoords = np.concatenate(xyData)
    energy = np.concatenate(ens)
    if ens_wave:
        energy = Constants.convert((energy - min(energy)), "wavenumbers", to_AU=False)
    carts = np.concatenate(cats)
    # make sure sorted & place in dictionary
    idx = np.lexsort((scoords[:, 0], scoords[:, 1]))
    dataDict["Dipoles"] = dipoles[idx]
    dataDict["xyData"] = scoords[idx]
    dataDict["Energies"] = energy[idx]
    dataDict["Cartesians"] = carts[idx]
    return dataDict

def rotate(dataDict):
    if dataDict["DataName"].find("w1") == 0:
        centralO_atom = 0
        xAxis_atom = 1
        xyPlane_atom = 2
        inversion_atom = None
    else:
        centralO_atom = 3  # Donor Oxygen
        xAxis_atom = 0  # Acceptor Oxygen
        inversion_atom = 5  # Shared Proton
        xyPlane_atom = None
    all_coords = Constants.convert(dataDict["Cartesians"], "angstroms", to_AU=True)
    all_dips = dataDict["Dipoles"].reshape((len(all_coords), 1, 3))
    # shift to origin
    o_coords = all_coords - all_coords[:, np.newaxis, centralO_atom]
    o_dips = all_dips - all_coords[:, np.newaxis, centralO_atom]
    # rotation to x-axis
    r1_coords, r1_dips = rot1(o_coords, o_dips, xAxis_atom)
    if dataDict["DataName"].find("w2") == 0:
        # for dimer, rotated atom is planar so "inverter" makes entire z-axis 0..
        rot_coords = r1_coords
        rot_dips = r1_dips
    elif dataDict["DataName"].find("w1") == 0:
        # for monomer, rotate the third H to the xy-plane
        rot_coords, rot_dips = rot2(r1_coords, r1_dips, xyPlane_atom, outerO1=0, outerO2=1)
    else:
        rot_coords, rot_dips = inverter(r1_coords, r1_dips, inversion_atom)  # inversion of designated atom
    dipadedodas = rot_dips.reshape(len(all_coords), 3)
    data_name = dataDict["DataName"]
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_rotcoords_OHO.npy"),
            rot_coords)
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy"),
            dipadedodas)
    get_xyz(os.path.join(dataDict["MainDir"], f"{data_name}_rotcoords_OHO.xyz"),
            Constants.convert(rot_coords, "angstroms", to_AU=False), dataDict["AtomStr"])
    print("saved xyz")
    return rot_coords, dipadedodas  # bohr & debye

def plot_potential(dataDict):
    plt.rcParams.update({'font.size': 20})
    x = np.unique(dataDict["xyData"][:, 0])
    y = np.unique(dataDict["xyData"][:, 1])
    ens = dataDict["Energies"]
    ens[ens > 9000] = 9000
    pot = ens.reshape((len(x), len(y)))
    fig, ax = plt.subplots(figsize=(12, 12), dpi=216)
    ax.contourf(x, y*2, pot, levels=15, cmap="RdYlBu")
    # plt.plot(dataDict["xyData"][:, 0], dataDict["xyData"][:, 1]*2, "o", color="white")
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
    data_name = dataDict["DataName"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy")):
        dips = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy"))
    else:
        rot_coords, dips = rotate(dataDict)
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
            # fig = plt.figure(figsize=(14, 12), dpi=216)
            # ax = fig.add_subplot(projection="3d")
            v = np.linspace(-3.0, 3.0, 21, endpoint=True)
            # im = ax.plot_surface(x, y * 2, dipoles, cmap="RdYlBu", vmin=-3.0, vmax=3.0, antialiased=False)
            im = ax.contourf(x, y * 2, dipoles, v, cmap="RdYlBu", vmin=-3.0, vmax=3.0)  # , extend="both")
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

if __name__ == '__main__':
    logs = ["w2_ScanR4B.log", "w2_ScanR5B*.log", "w6_ScanR4B*.log", "w6_ScanR5B*.log", "w6a_ScanR4B*.log", "w6a_ScanR5B*.log", "w1_Scan*.log"]
    for l in logs:
        print(l)
        Ddict = pull_data(l)
        # plot_potential(Ddict)
        plot_dipoles(Ddict)

