python3 build_dl_plugin_so.py 
gfortran -c dlModule.f90 
gfortran -o fortran-infer-test -L  /home/admin/WRF/xlf/fortran-python-test/infer-example  -Wl,-rpath=/home/admin/WRF/xlf/fortran-python-test/infer-example  -lDL_inference_plugin   fortran-infer.f90 dlModule.o

