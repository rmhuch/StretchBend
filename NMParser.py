from McUtils.GaussianInterface import GaussianLogReader
import numpy as np
import os

def pull_block(logfile):
    """
    This function reads a log file and pulls out the full "Frequencies" block. (Harmonic Normal Modes)
    :return:
    :rtype:
    """
    with GaussianLogReader(logfile) as reader:
        parse = reader.parse("HarmonicFrequencies")
    raw_data = parse["HarmonicFrequencies"]
    return raw_data

def format_disps(block):
    """

    :param block:
    :type block:
    :return:
    :rtype:
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


if __name__ == '__main__':
    print('hi RSCH')
    docs = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MoleculeDir = os.path.join(docs, "stretch_bend", "tetramer_16", "cage")
    f1 = os.path.join(MoleculeDir, "w4c_Hw1.log")
    b = pull_block(f1)
    disps = format_disps(b)

