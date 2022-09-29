## Getting Started
* This repository contains python code that can:
  * handle/parse Gaussian `.fchk` and `.log` data
  * create an object for a given water cluster size/isotopologue
  * perform various analysis on the `ClusterObj`

## Update Log
* April 7th
  * created `BuildTetCage` and `AnalyzeOneWater` classes
  * streamlined/updated other scripts to interact with (or were absorbed by) the classes.
* April 20th
  * created `BuildMonomer` and `BuildDimer` classes
  * added `RunTest.py` as a place to build/test/collect data
  * Local Mode frequencies and intensitites by finite difference is working for all systems,
    Normal Mode frequencies are still a little up in the air...
  - At this point it is clear there is a deviation in the LM intensity when compared to the harmonic,
    for the tetramer, less in the dimer, and hardly in the monomer, so this deviation is possibly some
    way of the hydrogen bonding manifesting itself... although I am not sure to what degree yet.
* ... September 29th
  * Important note: the SB data from before today has some typos for the VPT2.. in the LM the bound H always has a 
    larger intensity than a free H but, in the VPT2 the opposite is true, because of the way the csv of data for the 
    `Plots` object was created (by me by hand) there were some discrepancies in the code. Although as of 
    `SBdata_Sept29.csv` this is all cleared up. 
