from McUtils.GaussianInterface import GaussianLogReader
import numpy as np
import os
import csv

def pull_block(logfile):
    """
    This function reads a log file and pulls out the full "Frequencies" block. (Harmonic Normal Modes)
    :return: Block of lines encompassing all harmonic Normal Mode info.
    :rtype: str
    """
    with GaussianLogReader(logfile) as reader:
        parse = reader.parse("HarmonicFrequencies")
    raw_data = parse["HarmonicFrequencies"]
    return raw_data

def format_freqs(block):
    """
    Formats Harmonic frequencies from log file into 1D array.
    :param block: Result of `pull_block`
    :type block: str
    :return: `freqs` n-modes X 1
    :rtype: np.array
    """
    lines = block.splitlines(False)
    freqs = []
    for l in lines:
        if "Frequencies" in l:
            freqs.append([float(x) for x in l.split()[2:]])
        else:
            pass
    freqs = np.array(freqs).flatten()
    return freqs

def format_disps(block):
    """
    Formats Harmonic Normal Mode Displacements from log file into a n-modes X n-atoms X 3 (X Y Z) array
    :param block: Result of `pull_block`
    :type block: str
    :return: `disps` n-modes X n-atoms X 3 (X Y Z)
    :rtype: np.array
    """
    lines = block.splitlines(False)
    disps = []
    disp_array1 = None
    disp_array2 = None
    disp_array3 = None
    in_modes = False
    for l in lines:
        if "Atom" in l:
            in_modes = True
            disp_array1 = []
            disp_array2 = []
            disp_array3 = []
        elif in_modes:  # check to make sure we are still capturing data
            split_line = l.split()
            if len(split_line) < 5:
                disps.append(disp_array1)
                if len(disp_array2) > 0:
                    disps.append(disp_array2)
                if len(disp_array3) > 0:
                    disps.append(disp_array3)
                in_modes = False  # means we have captured all displacements
            else:
                try:
                    int(split_line[0])
                except TypeError:
                    in_modes = False  # means we have captured all displacements
                else:
                    in_modes = True
            if in_modes:
                comps = [float(x) for x in l.split()[2:]]
                if len(comps) > 8:
                    disp_array3.append(comps[6:9])
                if len(comps) > 3:
                    disp_array2.append(comps[3:6])
                disp_array1.append(comps[:3])
    disps = np.array(disps)
    return disps

def calcMaxDisps(disps_array, freqs, filename):
    """
    Takes the stack of NM displacements, calculate the displacement magnitude then determine
    the top three displacements for each mode.
    :param disps_array:
    :type disps_array:
    :param freqs:
    :type freqs:
    :param filename:
    :type filename: str
    :return: ** saves a csv file to given `filename`
    :rtype: csv
    """
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # create header for csv
        header = [str(n+1) for n in np.arange(len(disps_array[0]))]
        header.insert(0, "Frequency")
        header.insert(0, "Mode")
        csvwriter.writerow(header)

        # write each row
        for i, mode in enumerate(disps_array):
            norm_disps = np.linalg.norm(mode, axis=1)
            row = list(norm_disps)
            row.insert(0, freqs[i])
            row.insert(0, i+1)
            csvwriter.writerow(row)


if __name__ == '__main__':
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    f1 = os.path.join(MoleculeDir, "w4c_oneH8.log")
    b = pull_block(f1)
    disps = format_disps(b)
    freqs = format_freqs(b)
    filename = os.path.join(MoleculeDir, "H8_NMdisps.csv")
    calcMaxDisps(disps, freqs, filename)

