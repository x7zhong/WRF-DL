This github is the code implementation corresponding to the article "WRF-ML v1.0: A Bridge between WRF v4.3 and Machine Learning Parameterizations and its Application to Atmospheric Radiative Transferr"

Please contact x7zhong@gmail.com if you have any questions about this code.

# Code description:

## Build_WRF
Contains the libraries necessary for building WRF.

### Build_WRF/LIBRARIES/dl-inference-plugin
Contains the library necessary for building and running WRF_DL

## dl_inference
Contains the files used for building the library in dl-inference-plugin, and the python script to run ML models within WRF_DL

### dl_inference/model
ML-based radiation emulators are saved.

## WRF
Contains the files that are different from the original WRF v4.3 files and used for implementing the WRF-ML coupler.
To run the WRF coupled with ML-based radiation schemes, you need to add files that did not exist or overwrite the existing WRF files.

## example
Contains the examples about how to call Python script from Fortran code.

# Instructions
 
## Setting environment variables 

add the path to the WRF ML inference library
for example:
##WRF DL INFERENCE LIB
export WRF_DL_INFER_DIR=$DIR/dl-inference-plugin
export LD_LIBRARY_PATH=$WRF_DL_INFER_DIR/lib:$LD_LIBRARY_PATH

## modifications to the WRF configure.wrf
After running the ./configure when compiling WRF, modify the WRF configure.wrf file as:
add -L${path_to_dl-inference-plugin/lib} and "-lDL_inference_plugin" following LIB_EXTERNAL
![image](https://user-images.githubusercontent.com/65062130/191700283-f77a6391-f235-4273-a781-2eeec20f4b92.png)

## ML-based parameterization
Add ML-based parameterization schemes' related modules in the main/depend.common file, e.g.:
![image](https://user-images.githubusercontent.com/65062130/191700851-6d6c7a49-fd67-4257-ae4f-7ecff99698f3.png)

## add infer_init in the main/wrf.F to initialize the services of ML model inference

## profiling mode (optional)
This part is used for profiling






