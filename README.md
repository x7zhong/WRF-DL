This github is the code implementation corresponding to the article "A Bridge between WRF and Deep Learning Parameterization and Application on Deep Learning Parameterization of Atmospheric Radiative Transfer"

The code description is as follows:

Build_WRF: contains the libraries necessary for building WRF
Build_WRF/LIBRARIES/dl-inference-plugin: contains the library necessary for building and running WRF_DL

dl_inference: contains the files used for building the library in dl-inference-plugin, and the python script to run DL models within WRF_DL
WRF: contains the files that are different from the original WRF files and used for implementing the WRF-DL coupler 
