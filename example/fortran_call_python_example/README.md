This illustrates how to call Python using CFFI step by step, and more details can be referred to https://www.noahbrenowitz.com/post/calling-fortran-from-python/. 

1. The Fortran code for the "hello world" example is saved in the file fortran-call-python.f90. Within the Fortran code, an interface for a C function hello_world is defined, and this function will be called in the last step. 

2. The Python script is saved in the file "build_plugin_so.py". This script is used to generate a dynamic library. Type the following:

python build_plugin_so.py

If this finished correctly, you would see files including my_plugin.c, my_plugin.o, my_plugin.h, and libplugin.so.

3. Compile the Fortran program using the following command:
gfortran -o helloword -L ./ -Wl,-rpath=./ -lplugin fortran-call-python.f90

If this worked well, we could run the executable using the following:
./helloword