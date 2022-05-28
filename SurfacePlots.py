import os
from Converter import Constants
from McUtils.GaussianInterface import GaussianLogReader
from McUtils.Plots import ListContourPlot
from Rotator import *

def pull_data(logfile):
    dataDict = dict()
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MainDir = os.path.join(docs, "stretch_bend", "RyanOctomer")
    dataDict["MainDir"] = MainDir
    logpath = os.path.join(MainDir, logfile)
    dataDict["Molecule"] = logfile[:2]
    dataDict["ScanCoords"] = logfile[7:-4]
    if dataDict["Molecule"] == "w2":
        dataDict["AtomStr"] = ["O", "H", "H", "O", "H", "H"]
    elif dataDict["Molecule"] == "w6":
        dataDict["AtomStr"] = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    elif dataDict["Molecule"] == "w8":
        dataDict["AtomStr"] = ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H",
                               "O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"]
    with GaussianLogReader(logpath) as reader:
        parse = reader.parse(("ScanEnergies", "DipoleMoments", "StandardCartesianCoordinates"))
    raw_dips = parse["DipoleMoments"]
    dataDict["Dipoles"] = np.array(list(raw_dips))
    header, raw_data = parse["ScanEnergies"]
    dataDict["xyData"] = np.column_stack((raw_data[:, 1], raw_data[:, 2]))
    dataDict["Energies"] = Constants.convert(raw_data[:, -1] - np.min(raw_data[:, -1]), "wavenumbers", to_AU=False)
    dataDict["Cartesians"] = parse["StandardCartesianCoordinates"][1]
    return dataDict

def rotate(dataDict):
    centralO_atom = 3  # Donor Oxygen
    xAxis_atom = 0  # Acceptor Oxygen
    inversion_atom = 5  # Shared Proton
    all_coords = Constants.convert(dataDict["Cartesians"], "angstroms", to_AU=True)
    all_dips = dataDict["Dipoles"].reshape((len(all_coords), 1, 3))
    # shift to origin
    o_coords = all_coords - all_coords[:, np.newaxis, centralO_atom]
    o_dips = all_dips - all_coords[:, np.newaxis, centralO_atom]
    # rotation to x-axis
    r1_coords, r1_dips = rot1(o_coords, o_dips, xAxis_atom)
    rot_coords, rot_dips = inverter(r1_coords, r1_dips, inversion_atom)  # inversion of designated atom
    dipadedodas = rot_dips.reshape(len(all_coords), 3)
    data_name = dataDict["Molecule"] + "_" + dataDict["ScanCoords"]
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_rotcoords_OHO.npy"),
            rot_coords)
    np.save(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy"),
            dipadedodas)
    get_xyz(os.path.join(dataDict["MainDir"], f"{data_name}_rotcoords_OHO.xyz"),
            Constants.convert(rot_coords, "angstroms", to_AU=False), dataDict["AtomStr"])
    print("saved xyz")
    return rot_coords, dipadedodas  # bohr & debye

def plot_potential(dataDict):
    opts = dict(
        plot_style=dict(cmap="viridis_r", levels=15),
        axes_labels=['$\mathrm{R_{OH}}$ ($\mathrm{\AA}$)',
                     'HOH (Degrees)'])
    xy = dataDict["xyData"]
    pot = dataDict["Energies"]
    obj = ListContourPlot(np.column_stack((xy[:, 0], xy[:, 1], pot)), **opts)
    obj.colorbar = {"graphics": obj.graphics}
    obj.show()

def plot_dipoles(dataDict):
    data_name = dataDict["Molecule"] + "_" + dataDict["ScanCoords"]
    if os.path.exists(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy")):
        dips = np.load(os.path.join(dataDict["MainDir"], f"{data_name}_rotdips_OHO.npy"))
    else:
        rot_coords, dips = rotate(dataDict)
    xy = dataDict["xyData"]
    comp = ['X', 'Y', 'Z']
    for i in np.arange(dips.shape[1]):
        opts = dict(
            plot_style=dict(cmap="viridis_r", levels=15),
            axes_labels=['$\mathrm{R_{OH}}$ ($\mathrm{\AA}$)', 'HOH (Degrees)'])
        obj = ListContourPlot(np.column_stack((xy[:, 0], xy[:, 1], dips[:, i])), **opts)
        obj.plot_label = f'{comp[i]}-Component of Dipole'
        obj.colorbar = {"graphics": obj.graphics}
        obj.show()

if __name__ == '__main__':
    w2dict = pull_data("w2_ScanH2B.log")
    plot_dipoles(w2dict)
